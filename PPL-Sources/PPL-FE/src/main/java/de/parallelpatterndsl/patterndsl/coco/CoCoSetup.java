package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTModule;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLCoCoChecker;

/**
 * A class that handles the creation of context conditions (CoCo).
 * Also checks if the AST conforms all CoCos.
 */
public class CoCoSetup {

    private PatternDSLCoCoChecker checker;

    public CoCoSetup(){
        checker = new PatternDSLCoCoChecker();
    }

    /**
     * Initializes the CoCo checker with all CoCos.
     */
    public void Init(){
        IndexAccessListExistsCoCo indexAccessListExistsCoCo = new IndexAccessListExistsCoCo();
        VariableExistsCoCo variableExistsCoCo = new VariableExistsCoCo();
        ReadOnlyPatternParameterCoCo readOnlyPatternParameterCoCo = new ReadOnlyPatternParameterCoCo();
        PatternReturnVariableExistsCoCo patternReturnVariableExistsCoCo = new PatternReturnVariableExistsCoCo();
        FunctionReturnTypeExistsCoCo functionReturnTypeExistsCoCo = new FunctionReturnTypeExistsCoCo();
        ReturnStatementExistsCoCo returnStatementExistsCoCo = new ReturnStatementExistsCoCo();
        ReductionStatementAtLastPositionCoCo reductionStatementAtLastPositionCoCo = new ReductionStatementAtLastPositionCoCo();
        ReadOnlyConstantsCoCo readOnlyConstantsCoCo = new ReadOnlyConstantsCoCo();
        ConstantsWithoutCallExpressionCoCo constantsWithoutCallExpressionCoCo = new ConstantsWithoutCallExpressionCoCo();
        RemoveConditionalExpressionCoCo removeConditionalExpressionCoCo = new RemoveConditionalExpressionCoCo();
        RemoveQualifiedExpressionCoCo removeQualifiedExpressionCoCo = new RemoveQualifiedExpressionCoCo();
        RepetitionRuleWithinPatternsCoCo repetitionRuleWithinPatternsCoCo = new RepetitionRuleWithinPatternsCoCo();
        DynamicProgrammingSyntaxCoCo dynamicProgrammingSyntaxCoCo = new DynamicProgrammingSyntaxCoCo();
        ReadOnlyGlobalVarsCoCo readOnlyGlobalVarsCoCo = new ReadOnlyGlobalVarsCoCo();
        ListExpressionsOnlyForInitializationCoCo listExpressionsOnlyForInitializationCoCo = new ListExpressionsOnlyForInitializationCoCo();
        NoForEachLoopAssignmentCoCo noForEachLoopAssignmentCoCo = new NoForEachLoopAssignmentCoCo();
        CorrectCallReferenceCoCo correctCallReferenceCoCo = new CorrectCallReferenceCoCo();
        CorrectPatternCallReferenceCoCo correctPatternCallReferenceCoCo = new CorrectPatternCallReferenceCoCo();
        Init_ListWithLiteralsCoCo init_listWithLiteralsCoCo = new Init_ListWithLiteralsCoCo();
        PredefinedFunctionParameterCountCoCo predefinedFunctionParameterCountCoCo = new PredefinedFunctionParameterCountCoCo();
        IncrementOnlyAssignmentCoCo incrementOnlyAssignmentCoCo = new IncrementOnlyAssignmentCoCo();
        IncrementOnlyAssignmentVariableCoCo incrementOnlyAssignmentVariableCoCo = new IncrementOnlyAssignmentVariableCoCo();
        ListsMustBeInitializedCoCo listsMustBeInitializedCoCo = new ListsMustBeInitializedCoCo();
        NoExitInPatternCoCo noExitInPatternCoCo = new NoExitInPatternCoCo();
        LoopSkipOnlyWithinLoopCoCo loopSkipOnlyWithinLoopCoCo = new LoopSkipOnlyWithinLoopCoCo();
        ShadowVariableExistsCoCo shadowVariableExistsCoCo = new ShadowVariableExistsCoCo();
        CannotOverwritePredefinedFunctionsCoCo cannotOverwritePredefinedFunctionsCoCo = new CannotOverwritePredefinedFunctionsCoCo();
        ForControlWarningCoCo forControlWarningCoCo = new ForControlWarningCoCo();
        DimensionToSmallCoCo dimensionToSmallCoCo = new DimensionToSmallCoCo();
        NoCopyInFrontendCoCo noCopyInFrontendCoCo = new NoCopyInFrontendCoCo();
        NoINDEXAsParameterCoCo noINDEXAsParameterCoCo = new NoINDEXAsParameterCoCo();
        NoINDEXAsVariableCoCo noINDEXAsVariableCoCo = new NoINDEXAsVariableCoCo();
        NoIndexAccessInPatternCallsCoCo noIndexAccessInPatternCallsCoCo = new NoIndexAccessInPatternCallsCoCo();
        NoIndexAccessInIndexAccessCoCo noIndexAccessInIndexAccessCoCo = new NoIndexAccessInIndexAccessCoCo();
        UnusedParametersNotAllowedCoCo unusedParametersNotAllowedCoCo = new UnusedParametersNotAllowedCoCo();
        ReturnWithoutCallCoCo returnWithoutCallCoCo = new ReturnWithoutCallCoCo();
        ReturnWithoutStackVarCoCo returnWithoutStackVarCoCo = new ReturnWithoutStackVarCoCo();
        NoGlobalsAsArgumentCoCo noGlobalsAsArgumentCoCo = new NoGlobalsAsArgumentCoCo();


        checker.addCoCo(variableExistsCoCo);
        checker.addCoCo(correctPatternCallReferenceCoCo);
        checker.addCoCo(correctCallReferenceCoCo);
        checker.addCoCo(indexAccessListExistsCoCo);
        checker.addCoCo(returnStatementExistsCoCo);
        checker.addCoCo(readOnlyConstantsCoCo);
        checker.addCoCo(functionReturnTypeExistsCoCo);
        /** removed since SimpleAssignment not part of Grammar
        checker.addCoCo(reductionStatementAtLastPositionCoCo);
         **/
        checker.addCoCo(patternReturnVariableExistsCoCo);
        checker.addCoCo(readOnlyPatternParameterCoCo);
        checker.addCoCo(noForEachLoopAssignmentCoCo);
        checker.addCoCo(listExpressionsOnlyForInitializationCoCo);
        checker.addCoCo(readOnlyGlobalVarsCoCo);
        checker.addCoCo(dynamicProgrammingSyntaxCoCo);
        checker.addCoCo(repetitionRuleWithinPatternsCoCo);
        /** NOt in Grammar -> Test not possible
        checker.addCoCo(removeQualifiedExpressionCoCo);
        **/

        checker.addCoCo(removeConditionalExpressionCoCo);
        checker.addCoCo(constantsWithoutCallExpressionCoCo);
        checker.addCoCo(init_listWithLiteralsCoCo);
        checker.addCoCo(predefinedFunctionParameterCountCoCo);
        checker.addCoCo(incrementOnlyAssignmentCoCo);
        checker.addCoCo(incrementOnlyAssignmentVariableCoCo);
        checker.addCoCo(listsMustBeInitializedCoCo);
        checker.addCoCo(noExitInPatternCoCo);
        checker.addCoCo(loopSkipOnlyWithinLoopCoCo);
        checker.addCoCo(shadowVariableExistsCoCo);
        checker.addCoCo(cannotOverwritePredefinedFunctionsCoCo);
        checker.addCoCo(forControlWarningCoCo);
        checker.addCoCo(dimensionToSmallCoCo);
        checker.addCoCo(noCopyInFrontendCoCo);
        checker.addCoCo(noINDEXAsParameterCoCo);
        checker.addCoCo(noINDEXAsVariableCoCo);
        checker.addCoCo(noIndexAccessInPatternCallsCoCo);
        checker.addCoCo(noIndexAccessInIndexAccessCoCo);
        checker.addCoCo(unusedParametersNotAllowedCoCo);
        checker.addCoCo(returnWithoutCallCoCo);
        //checker.addCoCo(returnWithoutStackVarCoCo);
        checker.addCoCo(noGlobalsAsArgumentCoCo);

    }

    /**
     * Checks if module conforms the CoCos in checker.
     * @param module
     */
    public void Check(ASTModule module) {
        checker.checkAll(module);
    }
}
