#include "hipSYCL/compiler/cbs/CBSIntrinsics.hpp"

#include "hipSYCL/compiler/cbs/SubCfgFormation.hpp"

#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>

#include <vector>

namespace hipsycl::compiler::cbs {

void CBSIntrinsic::vectorizeAllInstances(
    llvm::Function &F, std::vector<SubCFG> &SubCfgs,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> *cont) {
  for (auto *Intrinsic : getIntrinsic(F, getName())) {
    auto *V = vectorizeFunction(F, *Intrinsic, findSubCfg(SubCfgs, Intrinsic), cont);
    Intrinsic->replaceAllUsesWith(V);
    Intrinsic->eraseFromParent();
  }
}

llvm::Value *CBSIntrinsic::vectorizeFunction(
    llvm::Function &F, llvm::CallInst &Intrinsic, SubCFG &SubCfg,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> *cont) {
  llvm::IRBuilder Builder{F.getContext()};
  // First argument is always the data argument
  const auto *Op0 = Intrinsic.getOperand(0);

  auto *UniformBlock = SubCfg.getEntry();
  auto *TerminatorInstruction = UniformBlock->getTerminator();
  Builder.SetInsertPoint(TerminatorInstruction);

  std::pair<llvm::Value *, Shape> res{};
  if (auto [Storage, Type] = getOrCreateValue(Builder, SubCfg, Intrinsic, cont);
      Type == Shape::UNIFORM) {
    if (!llvm::dyn_cast<llvm::Constant>(Storage)) {
      Storage = Builder.CreateLoad(Intrinsic.getFunctionType()->getReturnType(), Storage);
    }
    auto *SGIterations = [&]() {
      auto *Size = SubCfg.getHI().InnerSize;
      auto *InnerDimIterationsLeft = Builder.CreateSub(Size, SubCfg.getHI().InnerPhiInd);
      auto *EnoughIterationsLeft =
          Builder.CreateICmpULE(Builder.getInt64(SGSize), InnerDimIterationsLeft);
#if INCOMPLETE_SGS_OPT
      auto *noIncompleteSgs =
          mergeGVLoadsInEntry(F, "no-incomplete-sgs", EnoughIterationsLeft->getType());
      EnoughIterationsLeft = Builder.CreateLogicalOr(noIncompleteSgs, EnoughIterationsLeft);
#endif
      return Builder.CreateSelect(EnoughIterationsLeft, Builder.getInt64(SGSize),
                                  InnerDimIterationsLeft);
    }();

    res = vectorizeUniformValue(Storage, Builder, Intrinsic, SGIterations);
  } else {
    auto *VType = llvm::VectorType::get(Op0->getType(), llvm::ElementCount::getFixed(SGSize));

    auto *Mask = [&]() {
      std::vector<llvm::Constant *> v{};
      {
        v.reserve(SGSize);
        for (auto i = 0ul; i < SGSize; ++i) {
          v.emplace_back(Builder.getInt64(i));
        }
      }
      auto *Add = llvm::ConstantVector::get(v);
      auto *splat = Builder.CreateVectorSplat(llvm::ElementCount::getFixed(SGSize),
                                              SubCfg.getHI().InnerPhiInd);
      auto *sAdd = Builder.CreateAdd(splat, Add);
      return Builder.CreateICmpULT(sAdd,
                                   Builder.CreateVectorSplat(llvm::ElementCount::getFixed(SGSize),
                                                             SubCfg.getHI().InnerSize));
    }();

    auto *VLoad = Builder.CreateMaskedLoad(VType, Storage, llvm::Align(), Mask,
                                           neutralElement(VType, Builder, Intrinsic));
    res = vectorizeValue(VLoad, Builder, Intrinsic);
  }

  if (res.second == Shape::UNIFORM) {
    return res.first;
  }

  Builder.SetInsertPoint(&Intrinsic);
  return extractElement(Builder, res.first, SubCfg.getHI().ContiguousIdx, Intrinsic);
}

std::pair<llvm::Value *, CBSIntrinsic::Shape> CBSIntrinsic::getOrCreateValue(
    llvm::IRBuilder<> &Builder, const SubCFG &SubCfg, const llvm::CallInst &Intrinsic,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> *cont) {
  if (auto *Constant = llvm::dyn_cast<llvm::Constant>(Intrinsic.getOperand(0))) {
    return {Constant, Shape::UNIFORM};
  }
  auto *Load = llvm::dyn_cast<llvm::LoadInst>(Intrinsic.getOperand(0));
  assert(Load && "Op0 must be load inst");
  // NOT UNIFORM
  if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(Load->getPointerOperand())) {
    auto *Storage = GEP->getPointerOperand();

    // IS sub-group arrayified alloca
    if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(Storage)) {
      auto *Size = llvm::dyn_cast<llvm::ConstantInt>(Alloca->getArraySize());
      assert(Size);
      assert(Size->getSExtValue() == SGSize);
      return {Storage, Shape::VARYING};
    } else {
      // IS work-group arrayified alloca
      assert(llvm::isa<llvm::Argument>(Storage));
      auto InitialWgIndex = mergeGVLoadsInEntry(
          *llvm::dyn_cast<llvm::Argument>(Storage)->getParent(), "__cont_idx_without_sg");
      return {Builder.CreateGEP(Intrinsic.getArgOperand(0)->getType(), Storage, {InitialWgIndex}),
              Shape::VARYING};
    }
  }
  // UNIFORM (SUB_GROUP_LOCAL)
  else if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(Load->getPointerOperand())) {
    // IS SUBGROUP LOCAL
    auto *Size = llvm::dyn_cast<llvm::ConstantInt>(Alloca->getArraySize());
    assert(Size);
    // assert(Size->getSExtValue() == SGSize);
    return {Alloca, Shape::UNIFORM};
  } else {
    assert(llvm::dyn_cast<llvm::Argument>(Load->getPointerOperand()));
    return {Load->getPointerOperand(), Shape::UNIFORM};
  }
  llvm::outs() << "ERROR\n";
  std::exit(1);
}

SubCFG &CBSIntrinsic::findSubCfg(std::vector<SubCFG> &SubCfgs, llvm::Instruction *I) {
  auto it = std::find_if(SubCfgs.begin(), SubCfgs.end(), [&](const SubCFG &subCfg) {
    return std::any_of(subCfg.getNewBlocks().begin(), subCfg.getNewBlocks().end(),
                       [&](const llvm::BasicBlock *BB) { return BB == I->getParent(); });
  });
  assert(it != SubCfgs.end());
  return *it;
}

llvm::SmallVector<llvm::CallInst *, 8> CBSIntrinsic::getIntrinsic(llvm::Function &F,
                                                                  std::string_view str) {
  llvm::SmallVector<llvm::CallInst *, 8> Intrinsics{};
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *CallInst = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (auto *CalledF = CallInst->getCalledFunction()) {
          if (CalledF->getName().contains(str))
            Intrinsics.emplace_back(CallInst);
        }
      }
    }
  }
  return Intrinsics;
}

std::string ReduceIntrinsic::getTypeStr(llvm::Type *Type) {
  std::string type_str;
  llvm::raw_string_ostream rso(type_str);
  Type->print(rso);
  return rso.str();
}
std::pair<llvm::Value *, CBSIntrinsic::Shape>
ReduceIntrinsic::vectorizeUniformValue(llvm::Value *Storage, llvm::IRBuilder<> &Builder,
                                       llvm::CallInst &Intrinsic,
                                       llvm::Value *NumberOfLoopIterationsLeft) {
  auto *Idx = llvm::dyn_cast<llvm::ConstantInt>(Intrinsic.getOperand(1));
  auto *Type = Storage->getType();
  assert(Idx && "Op must be constant int");
  const auto v = Idx->getSExtValue();
  const bool isInt = Type->isIntegerTy();
  // min
  if (v == 2) {
    return {Storage, Shape::UNIFORM};
  }
  // max
  if (v == 3) {
    return {Storage, Shape::UNIFORM};
  }

  if (v == 0) {
    // ADD
    if (isInt)
      return {Builder.CreateMul(Storage,
                                Builder.CreateIntCast(NumberOfLoopIterationsLeft, Type, false)),
              Shape::UNIFORM};
    return {Builder.CreateFMul(
                Storage, Builder.CreateUIToFP(NumberOfLoopIterationsLeft, Storage->getType())),
            Shape::UNIFORM};
  }
  if (v == 1) {
    auto M = Intrinsic.getParent()->getParent()->getParent();
    if (!isInt) {
      auto *Pow =
          llvm::Intrinsic::getDeclaration(M, llvm::Intrinsic::powi, {Type, Builder.getInt32Ty()});
      llvm::Value *result = Storage;
      llvm::SmallVector<llvm::Value *> Args{
          result, Builder.CreateIntCast(NumberOfLoopIterationsLeft, Builder.getInt32Ty(), false)};
      result = Builder.CreateCall(Pow, Args);
      return {result, Shape::UNIFORM};
    }
    // WTF LLVM does not have integer pow intrinsic only floating point
    auto *Pow = createPowFunction(M, Type);
    llvm::Value *result = Storage;
    llvm::SmallVector<llvm::Value *> Args{
        result, Builder.CreateIntCast(NumberOfLoopIterationsLeft, Builder.getInt32Ty(), false)};
    result = Builder.CreateCall(Pow, Args);
    return {result, Shape::UNIFORM};
  }

  assert(false);
  return {};
}

std::pair<llvm::Value *, CBSIntrinsic::Shape>
ReduceIntrinsic::vectorizeValue(llvm::Instruction *VLoad, llvm::IRBuilder<> &Builder,
                                llvm::CallInst &Intrinsic) {
  auto *Type = Intrinsic.getOperand(0)->getType();
  const auto *Idx = llvm::dyn_cast<llvm::ConstantInt>(Intrinsic.getOperand(1));
  assert(Idx && "Op must be constant int");
  assert(Type && "Must be integer type");
  const auto v = Idx->getSExtValue();
  if (Type->isIntegerTy()) {
    const bool isSigned = llvm::dyn_cast<llvm::IntegerType>(Type)->getSignBit() > 0;
    if (v == 0) {
      return {Builder.CreateAddReduce(VLoad), Shape::UNIFORM};
    } else if (v == 1) {
      return {Builder.CreateMulReduce(VLoad), Shape::UNIFORM};
    } else if (v == 2) {
      return {Builder.CreateIntMinReduce(VLoad, isSigned), Shape::UNIFORM};
    } else if (v == 3) {
      return {Builder.CreateIntMaxReduce(VLoad, isSigned), Shape::UNIFORM};
    }
  } else {
    assert(Type->isFloatingPointTy());
    if (v == 0) {
      return {Builder.CreateFAddReduce(llvm::ConstantFP::getNegativeZero(Type), VLoad),
              Shape::UNIFORM};
    } else if (v == 1) {
      return {Builder.CreateFMulReduce(llvm::ConstantFP::get(Type, 1.0), VLoad), Shape::UNIFORM};
    } else if (v == 2) {
      return {Builder.CreateFPMinReduce(VLoad), Shape::UNIFORM};
    } else if (v == 3) {
      return {Builder.CreateFPMaxReduce(VLoad), Shape::UNIFORM};
    }
  }
  assert(false);
  return {};
}

llvm::Value *ReduceIntrinsic::neutralElement(llvm::VectorType *Type, llvm::IRBuilder<> &Builder,
                                             llvm::CallInst &Intrinsic) const {
  auto Idx = llvm::dyn_cast<llvm::ConstantInt>(Intrinsic.getOperand(1))->getSExtValue();
  if (Type->getElementType()->isIntegerTy()) {
    if (Idx == 0) {
      return llvm::ConstantInt::get(Type, 0);
    } else if (Idx == 1) {
      return llvm::ConstantInt::get(Type, 1);
    } else if (Idx == 2) {
      auto BitWidth = Type->getElementType()->getIntegerBitWidth();
      auto IsSigned = llvm::dyn_cast<llvm::IntegerType>(Type->getElementType())->getSignBit() > 0;
      auto Integer =
          IsSigned ? llvm::APInt::getSignedMaxValue(BitWidth) : llvm::APInt::getMaxValue(BitWidth);
      return llvm::ConstantInt::get(Type, Integer);
    } else if (Idx == 3) {
      auto BitWidth = Type->getElementType()->getIntegerBitWidth();
      auto IsSigned = llvm::dyn_cast<llvm::IntegerType>(Type->getElementType())->getSignBit() > 0;
      auto Integer =
          IsSigned ? llvm::APInt::getSignedMinValue(BitWidth) : llvm::APInt::getMinValue(BitWidth);
      return llvm::ConstantInt::get(Type, Integer);
    }
  } else {
    assert(Type->getElementType()->isFloatingPointTy());
    if (Idx == 0) {
      return llvm::ConstantFP::get(Type, -0.0);
    } else if (Idx == 1) {
      return llvm::ConstantFP::get(Type, 1.0);
    } else if (Idx == 2) {
      return llvm::ConstantFP::get(Type, std::numeric_limits<double>::infinity());
    } else if (Idx == 3) {
      return llvm::ConstantFP::get(Type, -std::numeric_limits<double>::infinity());
    }
  }

  assert(false);
  return {};
  return llvm::ConstantInt::get(Type, 0);
}

llvm::Function *ReduceIntrinsic::createPowFunction(llvm::Module *module, llvm::Type *Type) {
  llvm::LLVMContext &context = module->getContext();
  llvm::IRBuilder<> builder(context);

  llvm::FunctionType *funcType = llvm::FunctionType::get(Type, {Type, builder.getInt32Ty()}, false);
  auto powFunction = llvm::dyn_cast<llvm::Function>(
      module->getOrInsertFunction("pow." + getTypeStr(Type), funcType).getCallee());

  // Create a basic block and set the insert point
  llvm::BasicBlock *entry = llvm::BasicBlock::Create(context, "entry", powFunction);
  builder.SetInsertPoint(entry);

  // Get function arguments
  auto args = powFunction->arg_begin();
  llvm::Value *base = args++;
  base->setName("base");
  llvm::Value *exponent = args++;
  exponent->setName("exponent");

  // Initialize loop variables
  llvm::AllocaInst *result = builder.CreateAlloca(Type, nullptr, "result");
  llvm::Value *counter = builder.CreateAlloca(exponent->getType(), nullptr, "counter");
  builder.CreateStore(builder.getIntN(exponent->getType()->getIntegerBitWidth(), 0), counter);
  builder.CreateStore(builder.getIntN(Type->getIntegerBitWidth(), 1), result);

  // Create loop blocks
  llvm::BasicBlock *loopBB = llvm::BasicBlock::Create(context, "loop", powFunction);
  llvm::BasicBlock *afterLoopBB = llvm::BasicBlock::Create(context, "afterloop", powFunction);

  // Branch to loop block
  builder.CreateBr(loopBB);
  builder.SetInsertPoint(loopBB);

  // Load counter value
  llvm::Value *counterValue = builder.CreateLoad(exponent->getType(), counter, "counterValue");

  // Loop body
  llvm::Value *nextCounter = builder.CreateAdd(
      counterValue, builder.getIntN(exponent->getType()->getIntegerBitWidth(), 1), "nextcounter");
  builder.CreateStore(nextCounter, counter);
  auto resultX =
      builder.CreateMul(builder.CreateLoad(result->getAllocatedType(), result), base, "result");
  builder.CreateStore(resultX, result);

  llvm::Value *cond = builder.CreateICmpULT(nextCounter, exponent, "loopcond");
  builder.CreateCondBr(cond, loopBB, afterLoopBB);

  builder.SetInsertPoint(afterLoopBB);

  builder.CreateRet(builder.CreateLoad(result->getAllocatedType(), result));

  return powFunction;
}

std::pair<llvm::Value *, CBSIntrinsic::Shape>
ShuffleIntrinsic::vectorizeValue(llvm::Instruction *VLoad, llvm::IRBuilder<> &Builder,
                                 llvm::CallInst &Intrinsic) {
  auto *Idx = Intrinsic.getOperand(1);
  if (auto *Op1V = llvm::dyn_cast<llvm::ConstantInt>(Idx)) {
    return {Builder.CreateExtractElement(VLoad, Op1V), Shape::UNIFORM};
  } else {
    return {VLoad, Shape::VARYING};
  }
}

llvm::Value *ShuffleIntrinsic::extractElement(llvm::IRBuilder<> &Builder, llvm::Value *Value,
                                              llvm::Value *SgInductionVariable,
                                              llvm::CallInst &Intrinsic) {
  auto *Op1 = Intrinsic.getOperand(1);
  return Builder.CreateExtractElement(Value, Op1);
}

std::pair<llvm::Value *, CBSIntrinsic::Shape>
ExtractIntrinsic::vectorizeUniformValue(llvm::Value *Storage, llvm::IRBuilder<> &Builder,
                                        llvm::CallInst &Intrinsic,
                                        llvm::Value *NumberOfLoopIterationsLeft) {
  return {Storage, Shape::UNIFORM};
}

std::pair<llvm::Value *, CBSIntrinsic::Shape>
ExtractIntrinsic::vectorizeValue(llvm::Instruction *VLoad, llvm::IRBuilder<> &Builder,
                                 llvm::CallInst &Intrinsic) {
  auto *Idx = Intrinsic.getOperand(1);
  return {Builder.CreateExtractElement(VLoad, Idx), Shape::UNIFORM};
}
} // namespace hipsycl::compiler::cbs
