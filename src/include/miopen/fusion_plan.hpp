
#ifndef MIOPEN_GUARD_MLOPEN_FUSION_PLAN_HPP
#define MIOPEN_GUARD_MLOPEN_FUSION_PLAN_HPP

#include <miopen/config.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <miopen/fusion.hpp>
#include <miopen/search_options.hpp>

#include <boost/optional.hpp>

namespace miopen {

namespace solver {
struct ConvSolution;
} // namespace solver

enum class Exec_Arg_Type_t
{
    Scalar,
    Input_Ptr,
    Output_Ptr,
    Pointer,
    Padding,
    Default
};

struct Exec_arg_t
{
    std::string key;
    Exec_Arg_Type_t type;
    int size;
    OpKernelArg val;
    Exec_arg_t(std::string k, Exec_Arg_Type_t t, int s)
        : key(std::move(k)), type(t), size(s), val(OpKernelArg(0))
    {
    }
    Exec_arg_t(std::string k, Exec_Arg_Type_t t, int s, OpKernelArg v)
        : key(std::move(k)), type(t), size(s), val(v)
    {
    }
};

//MIOPEN_INTERNALS_EXPORT 是一个宏，类声明说明符，用于控制类的导出和导入，特别是在构建共享库时。
struct FusionContext;
struct MIOPEN_INTERNALS_EXPORT FusionPlanDescriptor : miopenFusionPlanDescriptor
{
    // 两个构造函数
    FusionPlanDescriptor() {}  
    FusionPlanDescriptor(miopenFusionDirection_t dir, const TensorDescriptor& inDesc);
    bool isValid() const { return is_valid; };  //判断融合计划是否有效
    miopenStatus_t AddOp(std::shared_ptr<FusionOpDescriptor> desc); //添加操作到融合计划
    TensorDescriptor DeriveOutputDescriptor(); //推导出输出张量描述符
    miopenStatus_t GetWorkspaceSizeImmed(const Handle& handle,
                                         size_t& workSpaceSize,
                                         miopenConvFwdAlgorithm_t algo); //获取执行融合操作所需的工作空间大小
    miopenStatus_t Execute(const Handle& handle,
                           const TensorDescriptor& inputDesc,
                           ConstData_t input,
                           const TensorDescriptor& outputDesc,
                           Data_t output,
                           const OperatorArgs& op_args); //执行融合计划
    miopenStatus_t Compile(const Handle& handle); //编译融合计划，准备执行
    // 
    std::vector<Solution> Find(const Handle& handle,
                               const std::function<fusion::FusionInvokeParams()>& invoke_params,
                               const std::optional<FindOptions>& options = std::nullopt) const;
    friend std::ostream& operator<<(std::ostream& stream, const FusionPlanDescriptor& fpd);

    miopenStatus_t
    GetConvAlgos(int reqAlgoCount, int& retAlgoCount, miopenConvFwdAlgorithm_t* ptrAlgos);
    miopenStatus_t SetConvAlgo(miopenConvFwdAlgorithm_t algo);

    miopenStatus_t GetOp(int op_idx, std::shared_ptr<FusionOpDescriptor>& desc);

    std::string GetAlgorithmName(const Handle& handle);
    std::vector<std::shared_ptr<FusionOpDescriptor>> op_map;

    miopenFusionDirection_t fusion_dir;
    TensorDescriptor input_desc;
    TensorDescriptor output_desc;
    int op_count = 0;
    bool is_valid;
    FusionKernelSourceType kernel_source_type;
    bool fp_contains_bn;
    miopenDataType_t data_type;
    std::vector<Exec_arg_t> arg_list;
    std::vector<Invoker> invokers;
    std::optional<miopenConvFwdAlgorithm_t> conv_fwd_algo;
};

} // namespace miopen

MIOPEN_DEFINE_OBJECT(miopenFusionPlanDescriptor, miopen::FusionPlanDescriptor);

#endif
