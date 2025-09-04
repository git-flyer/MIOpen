#ifndef GUARD_MLOPEN_ALLOCATOR_HPP
#define GUARD_MLOPEN_ALLOCATOR_HPP

#include <cassert>

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/manage_ptr.hpp>
#include <miopen/miopen.h>

namespace miopen {

struct AllocatorDeleter
{   //一个函数指针
    miopenDeallocatorFunction deallocator;
    void* context;

    template <class T>
    void operator()(T* x) const
    {
        assert(deallocator != nullptr);
        if(x != nullptr)
        {
            deallocator(context, x);
        }
    }
};
struct Allocator
{   // allocator和deallocator都是一个函数指针
    miopenAllocatorFunction allocator;
    miopenDeallocatorFunction deallocator;
    void* context;
    // 定义一个类型别名，用 ManageDataPtr 作为指向 *Data_t 的智能指针
    using ManageDataPtr =
        std::unique_ptr<typename std::remove_pointer<Data_t>::type, AllocatorDeleter>;
    // Allocator()返回这个智能指针
    ManageDataPtr operator()(std::size_t n) const
    {
        assert(allocator != nullptr);
        assert(deallocator != nullptr);
        auto result = allocator(context, n);  //result是一个void*类型
        if(result == nullptr && n != 0)
        {
            MIOPEN_THROW("Custom allocator failed to allocate memory for buffer size " +
                         std::to_string(n) + ": ");
        }//返回指向这个分配结果的智能指针
        return ManageDataPtr{DataCast(result), AllocatorDeleter{deallocator, context}};
    }
};

} // namespace miopen

#endif
