package de.parallelpatterndsl.patterndsl.abstractPatternTree;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.FunctionNode;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.Nodes.Functions.MainNode;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;


import java.util.ArrayList;
import java.util.HashMap;

/**
 * A class that stores the abstract pattern tree and additional information.
 */
public class AbstractPatternTree {

    /**
     * The root node of the abstract pattern tree.
     */
    private MainNode root;

    /**
     * A lookup table containing all function definitions.
     */
    private static HashMap<String, FunctionNode> functionTable = new HashMap<>();

    /**
     * Stores if the extended shape visitor already was executed.
     * This is used to avoid generating the call shapes more than once.
     */
    private static boolean extendedVisitorFinished = false;

    /**
     * Global variable symbol table.
     */
    private HashMap<String, Data> globalVariableTable;

    /**
     * Stores the initial assignments to the global variables.
     */
    private ArrayList<IRLExpression> globalAssignments;

    /**
     * Instance of the APT.
     */
    private static AbstractPatternTree instance;

    public AbstractPatternTree(MainNode root, HashMap<String, Data> globalVariableTable, ArrayList<IRLExpression> globalAssignments) {
        this.root = root;
        this.globalVariableTable = globalVariableTable;
        this.globalAssignments = globalAssignments;
        instance = this;
    }

    public static void setFunctionTable(HashMap<String, FunctionNode> functionTable) {
        AbstractPatternTree.functionTable = functionTable;
    }

    public HashMap<String, Data> getGlobalVariableTable() {
        return globalVariableTable;
    }

    public MainNode getRoot() {
        return root;
    }

    public static HashMap<String, FunctionNode> getFunctionTable() {
        return functionTable;
    }

    public static boolean isExtendedVisitorFinished() {
        return extendedVisitorFinished;
    }

    public static void setExtendedVisitorFinished(boolean extendedVisitorFinished) {
        AbstractPatternTree.extendedVisitorFinished = extendedVisitorFinished;
    }

    public ArrayList<IRLExpression> getGlobalAssignments() {
        return globalAssignments;
    }

    public void setGlobalAssignments(ArrayList<IRLExpression> globalAssignments) {
        this.globalAssignments = globalAssignments;
    }

    public static AbstractPatternTree getInstance() {
        return instance;
    }
}
