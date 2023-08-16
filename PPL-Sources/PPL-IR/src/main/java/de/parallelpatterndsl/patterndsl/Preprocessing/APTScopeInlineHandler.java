package de.parallelpatterndsl.patterndsl.Preprocessing;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.AbstractPatternTree;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.SerialNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.*;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Visitor.ExtendedShapeAPTVisitor;
import de.parallelpatterndsl.patterndsl.expressions.AssignmentExpression;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.expressions.OperationExpression;
import de.parallelpatterndsl.patterndsl.expressions.Operator;
import de.parallelpatterndsl.patterndsl.helperLibrary.RandomStringGenerator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.stream.Collectors;

/**
 * This class inlines function calls which contain parallel patterns.
 */
public class APTScopeInlineHandler implements ExtendedShapeAPTVisitor {

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

    public APTScopeInlineHandler(AbstractPatternTree APT) {
        this.APT = APT;
        currentVariableTable = new HashMap<>(APT.getRoot().getVariableTable());
    }

    @Override
    public void traverse(CallNode node) {

    }


    public void traverse(SerialNode node){
        for (PatternNode child: node.getChildren()) {
            child.accept(getRealThis());
            if (child instanceof BranchNode) {
                if (!((BranchNode) child).isInlined()) {
                    inlineCopy.add(child);
                }
            } else {
                inlineCopy.add(child);
            }
        }
    }

    @Override
    public void traverse(ReturnNode node) {

    }

    @Override
    public void traverse(ForEachLoopNode node) {

    }

    @Override
    public void traverse(ForLoopNode node) {

    }

    @Override
    public void traverse(ParallelCallNode node) {

    }

    @Override
    public void traverse(WhileLoopNode node) {

    }

    @Override
    public void traverse(BranchNode node) {
        if (node.getChildren().size() == 1) {
            BranchCaseNode branch = (BranchCaseNode) node.getChildren().get(0);
            if (branch.isHasCondition()) {
                if (branch.getChildren().get(0) instanceof ComplexExpressionNode) {
                    OperationExpression expression = (OperationExpression) ((ComplexExpressionNode) branch.getChildren().get(0)).getExpression();
                    if (expression.getOperators().isEmpty() && expression.getOperands().size() == 1) {
                        Data condition = expression.getOperands().get(0);
                        if (condition instanceof LiteralData) {
                            if (((LiteralData) condition).getValue() instanceof Boolean) {
                                if (((Boolean) ((LiteralData) condition).getValue())) {
                                    hasInlined = true;
                                    String previousID = inlineIdentifier;

                                    inlineIdentifier = RandomStringGenerator.getAlphaNumericString();

                                    node.setInlined(true);

                                    for (Data data: branch.getVariableTable().values().stream().filter(x -> (!(node.getVariableTable().values().contains(x))) ).collect(Collectors.toSet()) ) {
                                        Data newData = data.createInlineCopy(inlineIdentifier);
                                        VariableReplacementStack.getCurrentTable().put(data, newData);
                                        currentVariableTable.put(newData.getIdentifier(), newData);
                                    }

                                    for (int i = 1; i < branch.getChildren().size(); i++) {
                                        PatternNode child = branch.getChildren().get(i);
                                        if (child instanceof BranchNode) {
                                            child.accept(getRealThis());
                                            if (!((BranchNode) child).isInlined()) {
                                                inlineCopy.add(createInlineCopy(new ArrayList<>(APT.getGlobalVariableTable().values()), inlineIdentifier, currentVariableTable, child, APT.getRoot()));
                                            }
                                        } else {
                                            inlineCopy.add(createInlineCopy(new ArrayList<>(APT.getGlobalVariableTable().values()), inlineIdentifier, currentVariableTable, child, APT.getRoot()));
                                        }

                                    }
                                    APT.getRoot().setVariableTable(mergeVariableTable(APT.getRoot().getVariableTable(), currentVariableTable));
                                    inlineIdentifier = previousID;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private PatternNode createInlineCopy(ArrayList<Data> globalVars, String inlineIdentifier, HashMap<String, Data> variableTable, PatternNode original, PatternNode parent) {
        for (Data data: original.getVariableTable().values() ) {
            if (!VariableReplacementStack.getCurrentTable().containsKey(data)) {
                String newName = data.getIdentifier() + "_" + inlineIdentifier;
                Data newData = data.createInlineCopy(inlineIdentifier);
                currentVariableTable.put(newName, newData);
                VariableReplacementStack.getCurrentTable().put(data, newData);
                APT.getRoot().getVariableTable().put(newName, newData);
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
            currentVariableTable.put(((ParallelCallNode) result).getCallExpression().getIdentifier(),((ParallelCallNode) result).getCallExpression() );
        } else if (original instanceof ReturnNode) {
            ArrayList<ArrayData> closingScope = new ArrayList<>();
            for (Data data: currentVariableTable.values() ) {
                if (data instanceof ArrayData && !globalVars.contains(data)) {
                    if (!((ArrayData) data).isOnStack()) {
                        closingScope.add((ArrayData) data);
                    }
                }
            }
            result = new JumpStatementNode(closingScope, (ComplexExpressionNode) original.getChildren().get(0), "STOP_LABEL_" + inlineIdentifier, currentVariableTable.get("inlineReturn_" + inlineIdentifier));
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
            ((CallNode) result).setCallExpression(((CallNode) original).getCallExpression());
            ((CallNode) result).getCallExpression().createInlineCopies(globalVars,inlineIdentifier,currentVariableTable);
        }

        result.setParent(parent);
        result.setVariableTable(currentVariableTable);

        if (original instanceof CallNode && !(original instanceof ParallelCallNode)) {
            result.setChildren(new ArrayList<>());
        } else {
            ArrayList<PatternNode> children = new ArrayList<>();

            if (original instanceof ParallelCallNode) {
                children.add(createInlineCopy(globalVars, inlineIdentifier, currentVariableTable, original.getChildren().get(0), result));
            } else {
                for (PatternNode child : original.getChildren()) {
                    children.add(createInlineCopy(globalVars, inlineIdentifier, currentVariableTable, child, result));
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
