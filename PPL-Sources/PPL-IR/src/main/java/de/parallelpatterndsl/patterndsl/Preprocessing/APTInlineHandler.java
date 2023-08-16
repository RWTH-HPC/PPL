package de.parallelpatterndsl.patterndsl.Preprocessing;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
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
import de.se_rwth.commons.logging.Log;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * This class inlines function calls which contain parallel patterns.
 */
public class APTInlineHandler implements ExtendedShapeAPTVisitor {

    private AbstractPatternTree APT;

    private HashMap<String, Data> currentVariableTable;

    private ArrayList<PatternNode> inlineCopy = new ArrayList<>();

    private String inlineIdentifier = "";

    private boolean hasInlined = false;

    public boolean generateInlining() {
        HashMap<Data, Data> originalReplacements = new HashMap<>();
        APT.getRoot().getVariableTable().values().stream().filter(x -> (x instanceof ArrayData || x instanceof PrimitiveData)).forEach(x -> originalReplacements.put(x, x));

        VariableReplacementStack.addTable(originalReplacements);
        APT.getRoot().accept(this.getRealThis());
        APT.getRoot().setChildren(inlineCopy);
        APT.getRoot().setVariableTable(mergeVariableTable(currentVariableTable, APT.getRoot().getVariableTable()));
        VariableReplacementStack.removeLastTable();
        return hasInlined;
    }

    public APTInlineHandler(AbstractPatternTree APT) {
        this.APT = APT;
        currentVariableTable = new HashMap<>(APT.getRoot().getVariableTable());
    }

    @Override
    public void traverse(CallNode node) {

        if (AbstractPatternTree.getFunctionTable().get((node).getFunctionIdentifier()).isHasParallelDescendants() && !hasInlined) {
            hasInlined = true;
            HashMap<Data, Data> originalReplacements = new HashMap<>();
            APT.getGlobalVariableTable().values().stream().filter(x -> (x instanceof ArrayData || x instanceof PrimitiveData)).forEach(x -> originalReplacements.put(x, x));

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

            currentVariableTable.put("inlineReturn_" + inlineIdentifier, returnValue);

            APT.getRoot().getVariableTable().put(returnValue.getIdentifier(), returnValue);


            for (int i = 0; i < function.getArgumentValues().size(); i++) {
                Data parameter = function.getArgumentValues().get(i);
                Data parameterCopy;
                if (parameter instanceof ArrayData) {
                    parameterCopy = new ArrayData(parameter.getIdentifier() + "_" + inlineIdentifier, parameter.getTypeName(), false, ((ArrayData) parameter).getShape(), false);
                } else {
                    parameterCopy = new PrimitiveData(parameter.getIdentifier() + "_" + inlineIdentifier, parameter.getTypeName(), false);
                }
                parameterCopy.setInlinedParameter(true);
                APT.getRoot().getVariableTable().put(parameterCopy.getIdentifier(), parameterCopy);
                currentVariableTable.put(parameterCopy.getIdentifier(), parameterCopy);
                originalReplacements.put(parameter, parameterCopy);

                AssignmentExpression assignmentExpression;
                if (parameter instanceof ArrayData) {
                    //Create copied array
                    OperationExpression original = node.getArgumentExpressions().get(i);
                    ArrayList<Data> operands = new ArrayList<>();
                    operands.add(new FunctionReturnData("copy", PrimitiveDataTypes.COMPLEX_TYPE));

                    for (Data data : original.getOperands() ) {
                        if (data instanceof ArrayData || data instanceof PrimitiveData) {
                            Data input =  VariableReplacementStack.getCurrentTable().get(data);
                            operands.add(input);
                        } else {
                            operands.add(data);
                        }
                    }

                    ArrayList<Operator> operators = new ArrayList<>();
                    operators.add(Operator.LEFT_CALL_PARENTHESIS);
                    operators.addAll(node.getArgumentExpressions().get(i).getOperators());
                    operators.add(Operator.COMMA);
                    operands.add(new LiteralData<Integer>("1", PrimitiveDataTypes.INTEGER_64BIT, 1));
                    for (int dimension : original.getShape()) {
                        operators.add(Operator.MULTIPLICATION);
                        operands.add(new LiteralData<Integer>("dim", PrimitiveDataTypes.INTEGER_64BIT, dimension));
                    }
                    operators.add(Operator.RIGHT_CALL_PARENTHESIS);

                    OperationExpression functionCall = new OperationExpression(operands, operators);

                    FunctionInlineData functionInlineData = new FunctionInlineData("copy_" + RandomStringGenerator.getAlphaNumericString(), PrimitiveDataTypes.COMPLEX_TYPE, functionCall, 1);

                    ArrayList<Data> finalOperands = new ArrayList<>();
                    finalOperands.add(functionInlineData);


                    OperationExpression operationExpression = new OperationExpression(finalOperands, new ArrayList<>());

                    assignmentExpression = new AssignmentExpression(parameterCopy, new ArrayList<>(), operationExpression, Operator.ASSIGNMENT);
                } else {
                    assignmentExpression = new AssignmentExpression(parameterCopy, new ArrayList<>(), node.getArgumentExpressions().get(i).createInlineCopy(new ArrayList<>( APT.getGlobalVariableTable().values()), inlineIdentifier, currentVariableTable), Operator.ASSIGNMENT);
                }

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
            if (((AssignmentExpression) parent.getExpression()).getRhsExpression().getOperands().size() == 1) {
                ((AssignmentExpression) parent.getExpression()).getRhsExpression().getOperands().set(0,returnValue);
            }
            parent.getExpression().replaceDataElement(node.getCallExpression(), returnValue);
            parent.setChildren(new ArrayList<>());

            //Handle new replacements
            VariableReplacementStack.addTable(originalReplacements);

            for (PatternNode child: node.getChildren() ) {
                child.accept(getRealThis());
                if (!(child instanceof ReturnNode)) {
                    PatternNode copyCall = createInlineCopy(new ArrayList<>( APT.getGlobalVariableTable().values()), inlineIdentifier, currentVariableTable, child, APT.getRoot());
                    inlineCopy.add(copyCall);
                }
            }
            VariableReplacementStack.removeLastTable();

            JumpLabelNode jumpLabelNode = new JumpLabelNode(inlineIdentifier);
            jumpLabelNode.setVariableTable(currentVariableTable);
            jumpLabelNode.setParent(node.getParent());
            inlineCopy.add(jumpLabelNode);

            // add the inlined data to the parent node to allow the data trace generation.
            APT.getRoot().setVariableTable(mergeVariableTable(APT.getRoot().getVariableTable(), currentVariableTable));

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
    public void traverse(ParallelCallNode node) {

    }

    @Override
    public void traverse(ReturnNode node) {

        ArrayList<ArrayData> closingScope = new ArrayList<>();
        for (Data data: node.getVariableTable().values()) {
            if (data instanceof ArrayData && !APT.getGlobalVariableTable().containsValue(data)) {
                closingScope.add((ArrayData) VariableReplacementStack.getCurrentTable().get(data));
            }
        }

        if (node.getParent() != APT.getRoot()) {
            boolean doReplacement = false;
            Data callReplacer = null;
            if (!(((ComplexExpressionNode) node.getChildren().get(0)).getChildren().isEmpty())) {
                if (node.getChildren().get(0).getChildren().get(0) instanceof CallNode) {
                    if (node.doesReturnArray()) {
                        callReplacer = new ArrayData("callReplacer_"+RandomStringGenerator.getAlphaNumericString(), node.getFunctionType(), false, true, ((ComplexExpressionNode) node.getChildren().get(0)).getExpression().getShape(),false);
                    } else {
                        callReplacer = new PrimitiveData("callReplacer_"+RandomStringGenerator.getAlphaNumericString(), node.getFunctionType(), false, true);
                    }
                    currentVariableTable.put(callReplacer.getIdentifier(), callReplacer);
                    ComplexExpressionNode copyOfOriginal = (ComplexExpressionNode) createInlineCopy(new ArrayList<>( APT.getGlobalVariableTable().values()), inlineIdentifier, currentVariableTable, node.getChildren().get(0), APT.getRoot());
                    AssignmentExpression assignment = new AssignmentExpression(callReplacer, new ArrayList<>(), (OperationExpression) copyOfOriginal.getExpression(), Operator.ASSIGNMENT);
                    ComplexExpressionNode replacer = new ComplexExpressionNode(assignment);
                    replacer.setChildren(copyOfOriginal.getChildren());
                    replacer.setVariableTable(currentVariableTable);
                    replacer.setParent(APT.getRoot());
                    inlineCopy.add(replacer);
                    doReplacement = true;
                }
            }
            ComplexExpressionNode jumpExpression;
            if (doReplacement) {
                if (callReplacer == null) {
                    Log.error("Call replacement failed!");
                }
                ArrayList<Data> operands = new ArrayList<>();
                operands.add(callReplacer);
                OperationExpression op = new OperationExpression(operands, new ArrayList<>());
                jumpExpression = new ComplexExpressionNode(op);
            } else {
                jumpExpression = (ComplexExpressionNode) createInlineCopy(new ArrayList<>( APT.getGlobalVariableTable().values()), inlineIdentifier, currentVariableTable, node.getChildren().get(0), APT.getRoot());
            }
            jumpExpression.setVariableTable(currentVariableTable);
            jumpExpression.setChildren(new ArrayList<>());

            JumpStatementNode jumpStatementNode = new JumpStatementNode(closingScope, jumpExpression, "STOP_LABEL_" + inlineIdentifier, currentVariableTable.get("inlineReturn_" + inlineIdentifier));
            jumpStatementNode.setVariableTable(currentVariableTable);
            jumpStatementNode.setParent(APT.getRoot());
            jumpExpression.setParent(jumpStatementNode);
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
    public void traverse(BranchNode node) {

    }



    private PatternNode createInlineCopy(ArrayList<Data> globalVars, String inlineIdentifier, HashMap<String, Data> variableTable, PatternNode original, PatternNode parent) {
        HashMap<String, Data> localVariableTable;
        if (original instanceof LoopNode || original instanceof BranchCaseNode) {
            localVariableTable = new HashMap<>(variableTable);
        } else {
            localVariableTable = variableTable;
        }
        for (Data data: original.getVariableTable().values() ) {
            if (!VariableReplacementStack.getCurrentTable().containsKey(data)) {
                if ((original instanceof LoopNode) || (original instanceof BranchCaseNode)) {
                    VariableReplacementStack.addTable(VariableReplacementStack.getCurrentTable());
                }
                    String newName = data.getIdentifier() + "_" + inlineIdentifier;
                    Data newData = data.createInlineCopy(inlineIdentifier);
                localVariableTable.put(newName, newData);
                    VariableReplacementStack.getCurrentTable().put(data, newData);
                    //APT.getRoot().getVariableTable().put(newName, newData);
            }
        }
        PatternNode result;
        if (original instanceof BranchCaseNode) {
            result = new BranchCaseNode(((BranchCaseNode) original).isHasCondition());
        } else if (original instanceof BranchNode) {
            result = new BranchNode();
        } else if (original instanceof ComplexExpressionNode) {
            result = new ComplexExpressionNode(((ComplexExpressionNode) original).getExpression().createInlineCopy(globalVars,inlineIdentifier,localVariableTable));
        } else if (original instanceof ForEachLoopNode) {
            result = new ForEachLoopNode(localVariableTable.get(((ForEachLoopNode) original).getLoopControlVariable().getIdentifier()+ "_" + inlineIdentifier));
        } else if (original instanceof ForLoopNode) {
            result = new ForLoopNode(localVariableTable.get(((ForLoopNode) original).getLoopControlVariable().getIdentifier()+ "_" + inlineIdentifier));
        } else if (original instanceof ParallelCallNode) {
            result = new ParallelCallNode(((ParallelCallNode) original).getParameterCount(), ((ParallelCallNode) original).getFunctionIdentifier(), ((ParallelCallNode) original).getAdditionalArgumentCount());
            localVariableTable.put(((ParallelCallNode) result).getCallExpression().getIdentifier(),((ParallelCallNode) result).getCallExpression() );
            ((ParallelCallNode) result).setAdditionalArguments(new ArrayList<>(((ParallelCallNode) original).getAdditionalArguments()));
        } else if (original instanceof ReturnNode) {
            ArrayList<ArrayData> closingScope = new ArrayList<>();
            for (Data data: original.getVariableTable().values() ) {
                if (data instanceof ArrayData && !globalVars.contains(data) && ((OperationExpression) ((ComplexExpressionNode) original.getChildren().get(0)).getExpression()).getOperands().contains(data)) {
                    if (!((ArrayData) data).isOnStack()) {
                        closingScope.add((ArrayData) VariableReplacementStack.getCurrentTable().get(data));
                    }
                }
            }
            result = new JumpStatementNode(closingScope, (ComplexExpressionNode) original.getChildren().get(0), "STOP_LABEL_" + inlineIdentifier, localVariableTable.get("inlineReturn_" + inlineIdentifier));
            ComplexExpressionNode child = (ComplexExpressionNode) createInlineCopy(globalVars, inlineIdentifier, localVariableTable, (ComplexExpressionNode) original.getChildren().get(0), result);
            ((JumpStatementNode) result).setResultExpression(child);
        } else if (original instanceof SimpleExpressionBlockNode) {
            ArrayList<IRLExpression> expressions = new ArrayList<>();
            for (IRLExpression exp: ((SimpleExpressionBlockNode) original).getExpressionList() ) {
                expressions.add(exp.createInlineCopy(globalVars,inlineIdentifier,localVariableTable));
            }
            result = new SimpleExpressionBlockNode(expressions);
        } else if (original instanceof WhileLoopNode) {
            result = new WhileLoopNode();
        } else {
            result = new CallNode(((CallNode) original).getParameterCount(), ((CallNode) original).getFunctionIdentifier());
            FunctionInlineData callExpression = (FunctionInlineData) localVariableTable.get(((CallNode) original).getCallExpression().getIdentifier() + "_" + inlineIdentifier);
            callExpression.setCall(((CallNode) original).getCallExpression().getCall().createInlineCopy(globalVars, inlineIdentifier, localVariableTable));
            ((CallNode) result).setCallExpression(callExpression);
            if (parent instanceof ComplexExpressionNode) {
                ((ComplexExpressionNode) parent).getExpression().replaceDataElement(((CallNode) original).getCallExpression(), callExpression);
            }
        }

        result.setParent(parent);
        result.setVariableTable(localVariableTable);

        if (original instanceof CallNode && !(original instanceof ParallelCallNode)) {
            result.setChildren(new ArrayList<>());
        } else {
            ArrayList<PatternNode> children = new ArrayList<>();

            if (original instanceof ParallelCallNode) {
                children.add(createInlineCopy(globalVars, inlineIdentifier, localVariableTable, original.getChildren().get(0), result));
            } else {
                for (PatternNode child : original.getChildren()) {
                    children.add(createInlineCopy(globalVars, inlineIdentifier, localVariableTable, child, result));
                }
            }
            result.setChildren(children);
        }


        return result;

    }

    /**
     * Combines the two given Variable tables into a single one
     * @param first
     * @param second
     * @return
     */
    private HashMap<String, Data> mergeVariableTable(HashMap<String, Data> first, HashMap<String, Data> second) {
        HashMap<String, Data> result = new HashMap<>(first);
        for (Data entry: second.values() ) {
            if (!result.containsKey(entry.getIdentifier())) {
                result.put(entry.getIdentifier(), entry);
            }
        }
        return result;
    }

}
