package de.parallelpatterndsl.patterndsl.helperLibrary;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.DataAccess.DataAccess;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.FunctionInlineData;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.ParallelNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.PatternNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.CallNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Plain.ParallelCallNode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

/**
 * Helper for deep copy
 */
public class DeepCopyHelper {

    private static Stack<HashMap<String, Data>> scopeStack = new Stack<>();

    private static ArrayList<Data> variableTableDifference(Map<String, Data> outer, Map<String, Data> inner) {
        ArrayList<Data> result = new ArrayList<>();
        for (String testing: inner.keySet() ) {
            if (!outer.containsKey(testing)) {
                result.add(inner.get(testing));
            }
        }
        return result;
    }

    /**
     * Adds a new scope to the current stack
     * @param oldScope
     */
    public static void addScope(HashMap<String, Data> oldScope) {

        // Only holds for global vars
        if (scopeStack.isEmpty()) {
            scopeStack.push(oldScope);
            return;
        }

        ArrayList<Data> diff = variableTableDifference(scopeStack.peek(), oldScope);

        HashMap<String, Data> newScope = new HashMap<>(scopeStack.peek());

        ArrayList<FunctionInlineData> inlineToAdapt = new ArrayList<>();
        for ( Data toCopy: diff ) {
            newScope.put(toCopy.getIdentifier(), toCopy.deepCopy());
            if (toCopy instanceof FunctionInlineData) {
                inlineToAdapt.add((FunctionInlineData) toCopy);
            }
        }

        for (FunctionInlineData update: inlineToAdapt) {
            ((FunctionInlineData) newScope.get(update.getIdentifier())).updateOperands(update, newScope);
        }
        scopeStack.push(newScope);
    }

    /**
     * Adds a new scope to the current stack
     * @param oldScope
     */
    public static void addScope(HashMap<String, Data> oldScope, ArrayList<Data> parameters) {

        // Only holds for global vars
        if (scopeStack.isEmpty()) {
            scopeStack.push(oldScope);
            return;
        }

        ArrayList<Data> diff = variableTableDifference(scopeStack.peek(), oldScope);

        HashMap<String, Data> newScope = new HashMap<>(scopeStack.peek());

        ArrayList<FunctionInlineData> inlineToAdapt = new ArrayList<>();
        for ( Data toCopy: diff ) {
            newScope.put(toCopy.getIdentifier(), toCopy.deepCopy());
            if (toCopy instanceof FunctionInlineData) {
                inlineToAdapt.add((FunctionInlineData) toCopy);
            }
        }

        for (FunctionInlineData update: inlineToAdapt) {
            ((FunctionInlineData) newScope.get(update.getIdentifier())).updateOperands(update, newScope);
        }
        scopeStack.push(newScope);
    }

    /**
     * removes a scope from the current stack
     */
    public static void removeScope() {
        scopeStack.pop();
    }


    public static HashMap<String, Data> currentScope() {
        return scopeStack.peek();
    }

    /**
     * Updates the data trace (in-/output elements/accesses).
     * @param node
     */
    public static void DataTraceUpdate(PatternNode node) {
        // Handle input accesses
        ArrayList<DataAccess> inputAccesses = new ArrayList<>();
        for (DataAccess input: node.getInputAccesses() ) {
            inputAccesses.add(input.deepCopy());
        }
        node.setInputAccesses(inputAccesses);

        // Handle output accesses
        ArrayList<DataAccess> outputAccesses = new ArrayList<>();
        for (DataAccess output: node.getOutputAccesses() ) {
            outputAccesses.add(output.deepCopy());
        }
        node.setOutputAccesses(outputAccesses);

        // Handle input
        ArrayList<Data> input = new ArrayList<>();
        for (Data inputElements: node.getInputElements() ) {
            input.add(DeepCopyHelper.currentScope().get(inputElements.getIdentifier()));
        }
        node.setInputElements(input);

        // Handle output accesses
        ArrayList<Data> output = new ArrayList<>();
        for (Data outputElements: node.getOutputElements() ) {
            output.add(DeepCopyHelper.currentScope().get(outputElements.getIdentifier()));
        }
        node.setOutputElements(output);
    }


    /**
     * Defines the general setup for plain nodes
     * @param old
     * @param result
     */
    public static void basicSetup(PatternNode old, PatternNode result) {

        DeepCopyHelper.addScope(old.getVariableTable());
        result.setVariableTable(DeepCopyHelper.currentScope());

        DeepCopyHelper.DataTraceUpdate(result);

        ArrayList<PatternNode> newChildren = new ArrayList<>();
        for (PatternNode node: old.getChildren()) {
            PatternNode newNode = node.deepCopy();
            newChildren.add(newNode);
            newNode.setParent(result);
        }

        result.setChildren(newChildren);

        DeepCopyHelper.removeScope();
    }

    /**
     * Defines the general setup for plain nodes
     * @param old
     * @param result
     */
    public static void basicCallSetup(CallNode old, CallNode result) {

        DeepCopyHelper.addScope(old.getVariableTable());
        result.setVariableTable(DeepCopyHelper.currentScope());

        DeepCopyHelper.DataTraceUpdate(result);

        ArrayList<PatternNode> newChildren = new ArrayList<>();
        if (old instanceof ParallelCallNode) {
            PatternNode newNode = old.getChildren().get(0).deepCopy();
            newChildren.add(newNode);
            newNode.setParent(result);
        } else {
            result.setCallExpression((FunctionInlineData) DeepCopyHelper.currentScope().get(old.getCallExpression().getIdentifier()));
        }
        result.setChildren(newChildren);

        DeepCopyHelper.removeScope();
    }

    /**
     * Defines the general setup for function nodes.
     * @param old
     * @param result
     */
    public static void basicFunctionSetup(FunctionNode old, FunctionNode result) {

        DeepCopyHelper.addScope(old.getVariableTable(), old.getArgumentValues());
        result.setVariableTable(DeepCopyHelper.currentScope());

        result.setArgumentCount(old.getArgumentCount());


        for (Data parameter: old.getArgumentValues() ) {
            Data copied = parameter.deepCopy();
            result.addArgumentValues(copied);
            DeepCopyHelper.currentScope().put(copied.getIdentifier(), copied);
        }

        for (Data update: DeepCopyHelper.currentScope().values()) {
            if (update instanceof FunctionInlineData) {
                ((FunctionInlineData) DeepCopyHelper.currentScope().get(update.getIdentifier())).updateOperands((FunctionInlineData) update, DeepCopyHelper.currentScope());
            }
        }

        DeepCopyHelper.DataTraceUpdate(result);

        ArrayList<PatternNode> newChildren = new ArrayList<>();
        for (PatternNode node: old.getChildren()) {
            PatternNode newNode = node.deepCopy();
            newChildren.add(newNode);
            newNode.setParent(result);
        }

        result.setChildren(newChildren);

        DeepCopyHelper.removeScope();
    }

    /**
     * Defines the general setup for parallel function nodes.
     * @param old
     * @param result
     */
    public static void basicPatternSetup(ParallelNode old, ParallelNode result) {

        DeepCopyHelper.addScope(old.getVariableTable());
        result.setVariableTable(DeepCopyHelper.currentScope());

        result.setArgumentCount(old.getArgumentCount());

        for (Data parameter: old.getArgumentValues() ) {
            Data copied = parameter.deepCopy();
            result.addArgumentValues(copied);
            DeepCopyHelper.currentScope().put(copied.getIdentifier(), copied);
        }

        for (Data update: DeepCopyHelper.currentScope().values()) {
            if (update instanceof FunctionInlineData) {
                ((FunctionInlineData) DeepCopyHelper.currentScope().get(update.getIdentifier())).updateOperands((FunctionInlineData) update, DeepCopyHelper.currentScope());
            }
        }

        result.setReturnElement(DeepCopyHelper.currentScope().get(old.getReturnElement().getIdentifier()));

        DeepCopyHelper.DataTraceUpdate(result);

        ArrayList<PatternNode> newChildren = new ArrayList<>();
        for (PatternNode node: old.getChildren()) {
            PatternNode newNode = node.deepCopy();
            newChildren.add(newNode);
            newNode.setParent(result);
        }

        result.setChildren(newChildren);

        DeepCopyHelper.removeScope();
    }
}
