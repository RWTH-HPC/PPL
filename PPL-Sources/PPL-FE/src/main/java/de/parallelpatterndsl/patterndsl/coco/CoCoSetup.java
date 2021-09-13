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

        checker.addCoCo(correctPatternCallReferenceCoCo);
        checker.addCoCo(correctCallReferenceCoCo);
        checker.addCoCo(noForEachLoopAssignmentCoCo);
        checker.addCoCo(listExpressionsOnlyForInitializationCoCo);
        checker.addCoCo(readOnlyGlobalVarsCoCo);
        checker.addCoCo(dynamicProgrammingSyntaxCoCo);
        checker.addCoCo(repetitionRuleWithinPatternsCoCo);
        checker.addCoCo(removeQualifiedExpressionCoCo);
        checker.addCoCo(removeConditionalExpressionCoCo);
        checker.addCoCo(constantsWithoutCallExpressionCoCo);
        checker.addCoCo(readOnlyConstantsCoCo);
        checker.addCoCo(reductionStatementAtLastPositionCoCo);
        checker.addCoCo(returnStatementExistsCoCo);
        checker.addCoCo(functionReturnTypeExistsCoCo);
        checker.addCoCo(patternReturnVariableExistsCoCo);
        checker.addCoCo(readOnlyPatternParameterCoCo);
        checker.addCoCo(variableExistsCoCo);
        checker.addCoCo(indexAccessListExistsCoCo);
    }

    /**
     * Checks if module conforms the CoCos in checker.
     * @param module
     */
    public void Check(ASTModule module) {
        checker.checkAll(module);
    }
}
