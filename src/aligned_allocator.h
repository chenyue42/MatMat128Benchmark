#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

#include <cstdlib>
#include <stdexcept>
#include <cstddef>

template<typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;

    // Provide a rebind template.
    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept {}

    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        // Calculate total size in bytes.
        std::size_t total_size = n * sizeof(T);
        // std::aligned_alloc requires the total size to be a multiple of Alignment.
        if (total_size % Alignment != 0) {
            total_size = ((total_size / Alignment) + 1) * Alignment;
        }
        void* ptr = std::aligned_alloc(Alignment, total_size);
        if (!ptr)
            throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }
};

template<typename T, typename U, std::size_t Alignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) {
    return true;
}

template<typename T, typename U, std::size_t Alignment>
bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, Alignment>&) {
    return false;
}

#endif // ALIGNED_ALLOCATOR_H
