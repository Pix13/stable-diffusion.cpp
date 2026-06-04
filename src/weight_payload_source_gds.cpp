#ifdef SD_CUDA_GDS
#include "weight_payload_source_gds.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cufile.h>

// Dynamic cuFile function pointers. Signatures match the cuFile API:
//   CUfileError_t cuFileHandleRegister(CUfileHandle_t*, CUfileDescr_t*);
//   void          cuFileHandleDeregister(CUfileHandle_t);
//   CUfileError_t cuFileBufRegister(const void*, size_t, int);
//   CUfileError_t cuFileBufDeregister(const void*);
//   ssize_t       cuFileRead(CUfileHandle_t, void*, size_t, off_t file_off, off_t buf_off);
typedef CUfileError_t (*cuFileHandleRegister_t)(CUfileHandle_t*, CUfileDescr_t*);
typedef void (*cuFileHandleDeregister_t)(CUfileHandle_t);
typedef CUfileError_t (*cuFileBufRegister_t)(const void*, size_t, int);
typedef CUfileError_t (*cuFileBufDeregister_t)(const void*);
typedef ssize_t (*cuFileRead_t)(CUfileHandle_t, void*, size_t, off_t, off_t);

static cuFileHandleRegister_t   fn_cuFileHandleRegister   = nullptr;
static cuFileHandleDeregister_t fn_cuFileHandleDeregister = nullptr;
static cuFileBufRegister_t      fn_cuFileBufRegister      = nullptr;
static cuFileBufDeregister_t    fn_cuFileBufDeregister    = nullptr;
static cuFileRead_t             fn_cuFileRead             = nullptr;

static void* g_libcufile_handle = nullptr;

static bool load_cufile_symbols() {
    if (fn_cuFileRead) {
        return true;  // Already loaded.
    }
    g_libcufile_handle = dlopen("libcufile.so", RTLD_NOW | RTLD_GLOBAL);
    if (!g_libcufile_handle) {
        fprintf(stderr, "cuFile: libcufile.so not found (%s)\n", dlerror());
        return false;
    }

    fn_cuFileHandleRegister   = (cuFileHandleRegister_t)dlsym(g_libcufile_handle, "cuFileHandleRegister");
    fn_cuFileHandleDeregister = (cuFileHandleDeregister_t)dlsym(g_libcufile_handle, "cuFileHandleDeregister");
    fn_cuFileBufRegister      = (cuFileBufRegister_t)dlsym(g_libcufile_handle, "cuFileBufRegister");
    fn_cuFileBufDeregister    = (cuFileBufDeregister_t)dlsym(g_libcufile_handle, "cuFileBufDeregister");
    fn_cuFileRead             = (cuFileRead_t)dlsym(g_libcufile_handle, "cuFileRead");

    if (!fn_cuFileHandleRegister || !fn_cuFileHandleDeregister ||
        !fn_cuFileBufRegister || !fn_cuFileBufDeregister || !fn_cuFileRead) {
        fprintf(stderr, "cuFile: missing required symbols in libcufile.so\n");
        return false;
    }

    return true;
}

CudaGdsWeightSource::CudaGdsWeightSource(const std::string& path)
    : WeightPayloadSource(path),
      fd_(-1),
      cufile_handle_(nullptr),
      cufile_desc_(nullptr) {}

CudaGdsWeightSource::~CudaGdsWeightSource() {
    cleanup();
}

bool CudaGdsWeightSource::init_cufile() {
    return load_cufile_symbols();
}

bool CudaGdsWeightSource::register_file_handle() {
    cufile_desc_ = malloc(sizeof(CUfileDescr_t));
    if (!cufile_desc_) {
        fprintf(stderr, "CudaGdsWeightSource: failed to allocate CUfileDescr_t\n");
        return false;
    }
    memset(cufile_desc_, 0, sizeof(CUfileDescr_t));

    CUfileDescr_t* desc = static_cast<CUfileDescr_t*>(cufile_desc_);
    desc->type          = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    desc->handle.fd     = fd_;

    CUfileHandle_t* handle = static_cast<CUfileHandle_t*>(malloc(sizeof(CUfileHandle_t)));
    if (!handle) {
        fprintf(stderr, "CudaGdsWeightSource: failed to allocate CUfileHandle_t\n");
        free(cufile_desc_);
        cufile_desc_ = nullptr;
        return false;
    }

    CUfileError_t err = fn_cuFileHandleRegister(handle, desc);
    if (err.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "CudaGdsWeightSource: cuFileHandleRegister failed (err=%d)\n", err.err);
        free(handle);
        free(cufile_desc_);
        cufile_desc_ = nullptr;
        return false;
    }

    cufile_handle_ = handle;
    return true;
}

bool CudaGdsWeightSource::open() {
    if (!init_cufile()) {
        return false;
    }

    // Open with O_DIRECT for strict mode (bypass page cache). cuFile requires
    // O_DIRECT for the true GPUDirect Storage path; without it cuFile may fall
    // back to a compatibility path that routes through host memory.
    fd_ = ::open(source_path_.c_str(), O_RDONLY | O_DIRECT);
    if (fd_ < 0) {
        fprintf(stderr,
                "CudaGdsWeightSource: failed to open '%s' with O_DIRECT: %s\n",
                source_path_.c_str(), strerror(errno));
        return false;
    }

    if (!register_file_handle()) {
        close();
        return false;
    }

    return true;
}

bool CudaGdsWeightSource::register_buffer(void* device_ptr, size_t size) {
    if (!fn_cuFileBufRegister || !device_ptr || size == 0) {
        return false;
    }

    // Avoid duplicate registration.
    if (registered_buffers_.count(device_ptr)) {
        return true;  // Already registered.
    }

    CUfileError_t err = fn_cuFileBufRegister(device_ptr, size, 0);
    if (err.err != CU_FILE_SUCCESS) {
        fprintf(stderr, "CudaGdsWeightSource: cuFileBufRegister failed (err=%d)\n", err.err);
        return false;
    }

    registered_buffers_[device_ptr] = size;
    return true;
}

void CudaGdsWeightSource::deregister_all_buffers() {
    for (auto& [ptr, size] : registered_buffers_) {
        (void)size;
        if (fn_cuFileBufDeregister) {
            fn_cuFileBufDeregister(ptr);
        }
    }
    registered_buffers_.clear();
}

bool CudaGdsWeightSource::read_to_device(const WeightSpan& span,
                                         ggml_backend_buffer_t /*dst_buffer*/,
                                         void* dst_device_ptr,
                                         size_t dst_offset,
                                         void* /*backend_stream_or_null*/) {
    if (!fn_cuFileRead || !cufile_handle_ || fd_ < 0) {
        fprintf(stderr, "CudaGdsWeightSource: not initialized\n");
        return false;
    }

    CUfileHandle_t* handle = static_cast<CUfileHandle_t*>(cufile_handle_);

    // Use the aligned read range if available, otherwise fall back to the exact
    // offset/size. dst_device_ptr must be the base pointer registered with
    // cuFileBufRegister; dst_offset is the offset into that registered buffer.
    off_t  file_off   = static_cast<off_t>(span.aligned_read_bytes ? span.aligned_file_offset : span.file_offset);
    size_t read_bytes = span.aligned_read_bytes ? span.aligned_read_bytes : span.payload_bytes;

    ssize_t n = fn_cuFileRead(*handle,
                              dst_device_ptr,
                              read_bytes,
                              file_off,
                              static_cast<off_t>(dst_offset));

    if (n < 0) {
        fprintf(stderr,
                "CudaGdsWeightSource: cuFileRead failed for '%s': "
                "errno=%d, offset=%lld, bytes=%zu\n",
                span.tensor_name.c_str(), errno,
                static_cast<long long>(file_off), read_bytes);
        return false;
    }

    if (static_cast<size_t>(n) < span.payload_bytes) {
        fprintf(stderr,
                "CudaGdsWeightSource: short read for '%s': expected %zu, got %zd\n",
                span.tensor_name.c_str(), span.payload_bytes, n);
        return false;
    }

    // If the payload starts inside the aligned read range, the caller is
    // responsible for accounting payload_offset_inside_aligned_read when
    // binding the tensor to the device pointer; no host buffer is used here.
    return true;
}

void CudaGdsWeightSource::close() {
    cleanup();
}

void CudaGdsWeightSource::cleanup() {
    deregister_all_buffers();

    if (cufile_handle_) {
        CUfileHandle_t* handle = static_cast<CUfileHandle_t*>(cufile_handle_);
        if (fn_cuFileHandleDeregister) {
            fn_cuFileHandleDeregister(*handle);
        }
        free(handle);
        cufile_handle_ = nullptr;
    }

    if (cufile_desc_) {
        free(cufile_desc_);
        cufile_desc_ = nullptr;
    }

    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

std::unique_ptr<CudaGdsWeightSource> create_cuda_gds_source(const std::string& path) {
    auto source = std::make_unique<CudaGdsWeightSource>(path);
    if (!source->init_cufile()) {
        return nullptr;
    }
    return source;
}

#endif  // SD_CUDA_GDS
