package de.parallelpatterndsl.patterndsl;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.SerialNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.APTVisitor;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Optional;

/**
 * This class inlines function calls which contain parallel patterns.
 */
public class APTInlineHandler implements ExtendedShapeAPTVisitor {

    private AbstractPatternTree APT;

    private HashMap<String, Data> currentVariableTable = new HashMap<>();

    private ArrayList<PatternNode> inlineCopy = new ArrayList<>();

    private String inlineIdentifier = "";

    public void generateInlining() {
        APT.getRoot().accept(this.getRealThis());
        APT.getRoot().setChildren(inlineCopy);
    }

    public APTInlineHandler(AbstractPatternTree APT) {
        this.APT = APT;
    }

    @Override
    public void traverse(CallNode node) {

        if (AbstractPatternTree.getFunctionTable().get((node).getFunctionIdentifier()).isHasParallelDescendants()) {

            String previousID = inlineIdentifier;

            inlineIdentifier = RandomStringGenerator.getAlphaNumericString();
            node.getCallExpression().setAPTIsInlined(true);

            SerialNode function = (SerialNode) AbstractPatternTree.getFunctionTable().get((node).getFunctionIdentifier());

            Data returnValue;

            if (function.isList()) {
                returnValue = new ArrayData("inlineReturn_" + inlineIdentifier, function.getReturnType(), false, function.getShape(), false);
            } else {
                returnValue = new PrimitiveData("inlineReturn_" + inlineIdentifier, function.getReturnType(), false);
            }

            APT.getRoot().getVariableTable().put(returnValue.getIdentifier(), returnValue);

            for (int i = 0; i < function.getArgumentValues().size(); i++) {
                Data parameter = function.getArgumentValues().get(i);
                Data parameterCopy;
                if (parameter instanceof ArrayData) {
                    parameterCopy = new ArrayData(parameter.getIdentifier() + "_" + inlineIdentifier, parameter.getTypeName(), false, ((ArrayData) parameter).getShape(), false);
                } else {
                    parameterCopy = new PrimitiveData(parameter.getIdentifier() + "_" + inlineIdentifier, parameter.getTypeName(), false);
                }
                APT.getRoot().getVariableTable().put(parameterCopy.getIdentifier(), parameterCopy);

                //Create copied array
                OperationExpression original = node.getArgumentExpressions().get(i);
                ArrayList<Data> operands = new ArrayList<>();
                operands.add(new FunctionReturnData("copy", PrimitiveDataTypes.COMPLEX_TYPE));
                operands.addAll(original.getOperands());

                ArrayList<Operator> operators = new ArrayList<>();
                operators.add(Operator.LEFT_CALL_PARENTHESIS);
                operators.addAll(node.getArgumentExpressions().get(i).getOperators());
                operators.add(Operator.COMMA);
                operands.add(new LiteralData<Integer>("1",PrimitiveDataTypes.INTEGER_8BIT, 1));
                for (int dimension: original.getShape() ) {
                    operators.add(Operator.MULTIPLICATION);
                    operands.add(new LiteralData<Integer>("dim", PrimitiveDataTypes.INTEGER_64BIT, dimension));
                }
                operators.add(Operator.RIGHT_CALL_PARENTHESIS);

                OperationExpression functionCall = new OperationExpression(operands, operators);

                FunctionInlineData functionInlineData = new FunctionInlineData("copy_" + RandomStringGenerator.getAlphaNumericString(), PrimitiveDataTypes.COMPLEX_TYPE, functionCall, 1);

                ArrayList<Data> finalOperands = new ArrayList<>();
                finalOperands.add(functionInlineData);

                OperationExpression operationExpression = new OperationExpression(finalOperands, new ArrayList<>());

                AssignmentExpression assignmentExpression = new AssignmentExpression(parameterCopy,new ArrayList<>(), operationExpression, Operator.ASSIGNMENT );

                ComplexExpressionNode complexExpressionNode = new ComplexExpressionNode(assignmentExpression);

                complexExpressionNode.setParent(APT.getRoot());

                complexExpressionNode.setVariableTable(currentVariableTable);

                inlineCopy.add(complexExpressionNode);

                // generate children from call parameters
                ComplexExpressionNode parameters = (ComplexExpressionNode) node.getParent();
                if (!parameters.getChildren().isEmpty()) {
                    complexExpressionNode.setChildren(new ArrayList<>());
                } else {
                    ArrayList<PatternNode> children = new ArrayList<>();
                    ArrayList<Data> dataArrayList = new ArrayList<>(((AssignmentExpression)complexExpressionNode.getExpression()).getRhsExpression().getOperands());
                    while (!dataArrayList.isEmpty()) {
                        Data testing = dataArrayList.get(0);
                        if (testing instanceof FunctionInlineData) {
                            for (PatternNode potentialChild: parameters.getChildren() ) {
                                if (potentialChild instanceof CallNode) {
                                    if (((CallNode) potentialChild).getCallExpression() == testing) {
                                        children.add(potentialChild);
                                    }
                                }
                            }
                        }

                        dataArrayList.remove(0);
                    }
                    complexExpressionNode.setChildren(children);
                }

                //TODO: define the copy function
            }

            ComplexExpressionNode parent = (ComplexExpressionNode) node.getParent();

            parent.getExpression().replaceDataElement(node.getCallExpression(), returnValue);
            parent.setChildren(new ArrayList<>());

            for (PatternNode child: node.getChildren() ) {
                child.accept(getRealThis());
                if (!(child instanceof ReturnNode)) {
                    inlineCopy.add(createInlineCopy(new ArrayList<>( APT.getGlobalVariableTable().values()), inlineIdentifier, currentVariableTable, child, APT.getRoot()));
                }
            }

            JumpLabelNode jumpLabelNode = new JumpLabelNode(inlineIdentifier);
            inlineCopy.add(jumpLabelNode);

            currentVariableTable = new HashMap<>(APT.getGlobalVariableTable());
            inlineIdentifier = previousID;
        }




    }


    public void traverse(SerialNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
            inlineCopy.add(child);
        }
    }

    @Override
    public void traverse(ReturnNode node) {
        ArrayList<ArrayData> closingScope = new ArrayList<>();
        for (Data data: currentVariableTable.values()) {
            if (data instanceof ArrayData && !APT.getGlobalVariableTable().containsValue(data)) {
                closingScope.add((ArrayData) data);
            }
        }

        if (node.getParent() != APT.getRoot()) {
            JumpStatementNode jumpStatementNode = new JumpStatementNode(closingScope, (ComplexExpressionNode) node.getChildren().get(0), "STOP_LABEL" + inlineIdentifier, currentVariableTable.get("inlineReturn_" + inlineIdentifier));

            inlineCopy.add(jumpStatementNode);
        }
    }

    @Override
    public void traverse(ForEachLoopNode node) {

    }

    @Override
    public void traverse(ForLoopNode node) {

    }

    @Override
    public void traverse(WhileLoopNode node) {

    }

    @Override
    public void traverse(ParallelCallNode node) {

    }

    @Override
    public void traverse(BranchNode node) {

    }

    public PatternNode createInlineCopy(ArrayList<Data> globalVars, String inlineIdentifier, HashMap<String, Data> variableTable, PatternNode original, PatternNode parent) {
        for (Data data: original.getVariableTable().values() ) {
            if (!globalVars.contains(data) && variableTable.get(data.getIdentifier() + "_" + inlineIdentifier) == null) {
                currentVariableTable.put(data.getIdentifier() + "_" + inlineIdentifier, data.createInlineCopy(inlineIdentifier));
            }
        }
        PatternNode result;
        if (original instanceof BranchCaseNode) {
            result = new BranchCaseNode(((BranchCaseNode) original).isHasCondition());
        } else if (original instanceof BranchNode) {
            result = new BranchNode();
        } else if (original instanceof ComplexExpressionNode) {
            result = new ComplexExpressionNode(((ComplexExpressionNode) original).getExpression().createInlineCopy(globalVars,inlineIdentifier,currentVariableTable));
        } else if (original instanceof ForEachLoopNode) {
            result = new ForEachLoopNode(currentVariableTable.get(((ForEachLoopNode) original).getLoopControlVariable().getIdentifier()+ "_" + inlineIdentifier));
        } else if (original instanceof ForLoopNode) {
            result = new ForLoopNode(currentVariableTable.get(((ForLoopNode) original).getLoopControlVariable().getIdentifier()+ "_" + inlineIdentifier));
        } else if (original instanceof ParallelCallNode) {
            result = new ParallelCallNode(((ParallelCallNode) original).getParameterCount(), ((ParallelCallNode) original).getFunctionIdentifier(), ((ParallelCallNode) original).getAdditionalArgumentCount());
            ((ParallelCallNode) result).setAdditionalArguments(new ArrayList<>(((ParallelCallNode) original).getAdditionalArguments()));
        } else if (original instanceof ReturnNode) {
            ArrayList<ArrayData> closingScope = new ArrayList<>();
            for (Data data: currentVariableTable.values() ) {
                if (data instanceof ArrayData && !globalVars.contains(data)) {
                    if (!((ArrayData) data).isOnStack()) {
                        closingScope.add((ArrayData) data);
                    }
                }
            }
            result = new JumpStatementNode(closingScope, (ComplexExpressionNode) original.getChildren().get(0), "STOP_LABEL" + inlineIdentifier, currentVariableTable.get("inlineReturn_" + inlineIdentifier));
        } else if (original instanceof SimpleExpressionBlockNode) {
            ArrayList<IRLExpression> expressions = new ArrayList<>();
            for (IRLExpression exp: ((SimpleExpressionBlockNode) original).getExpressionList() ) {
                expressions.add(exp.createInlineCopy(globalVars,inlineIdentifier,currentVariableTable));
            }
            result = new SimpleExpressionBlockNode(expressions);
        } else if (original instanceof WhileLoopNode) {
            result = new WhileLoopNode();
        } else {
            result = new CallNode(((CallNode) original).getParameterCount(), ((CallNode) original).getFunctionIdentifier());
            ((CallNode) result).setCallExpression((FunctionInlineData) currentVariableTable.get(((CallNode) original).getCallExpression().getIdentifier() + "_" + inlineIdentifier));
        }

        result.setParent(parent);
        result.setVariableTable(currentVariableTable);

        if (original instanceof CallNode) {
            result.setChildren(new ArrayList<>());
        } else {
            ArrayList<PatternNode> children = new ArrayList<>();

            for (PatternNode child: original.getChildren() ) {
                children.add(createInlineCopy(globalVars,inlineIdentifier,currentVariableTable,child,result));
            }
            result.setChildren(children);
        }

        return result;

    }


}
