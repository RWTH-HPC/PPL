package de.parallelpatterndsl.patterndsl.MappingTree;

import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.Function.MainMapping;
import de.parallelpatterndsl.patterndsl.MappingTree.Nodes.FunctionMapping;
import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;
import de.parallelpatterndsl.patterndsl.expressions.IRLExpression;
import de.parallelpatterndsl.patterndsl.hardwareDescription.Device;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Class definition of the abstract mapping tree
 */
public class AbstractMappingTree {

    /**
     * Describes the root node of the AMT
     */
    private MainMapping root;

    /**
     * Stores the global assignment of names and variables
     */
    private HashMap<String, Data> GlobalVariableTable;

    /**
     * Stores the global assignment of names and function definitions
     */
    private static HashMap<String, FunctionMapping> functionTable;

    /**
     * Stores the device executing the main execution in distributed programming (e.g. rank 0 with all serial executions)
     */
    private static Device defaultDevice;

    /**
     * Stores the initial assignments to the global variables.
     */
    private ArrayList<IRLExpression> globalAssignments;


    public AbstractMappingTree(MainMapping root, HashMap<String, Data> globalVariableTable, ArrayList<IRLExpression> globalAssignments) {
        this.root = root;
        GlobalVariableTable = globalVariableTable;
        this.globalAssignments = globalAssignments;
    }

    public static HashMap<String, FunctionMapping> getFunctionTable() {
        return functionTable;
    }


    public static void setFunctionTable(HashMap<String, FunctionMapping> functionTable) {
        AbstractMappingTree.functionTable = functionTable;
    }

    public static Device getDefaultDevice() {
        return defaultDevice;
    }

    public static void setDefaultDevice(Device defaultDevice) {
        AbstractMappingTree.defaultDevice = defaultDevice;
    }

    public MainMapping getRoot() {
        return root;
    }

    public HashMap<String, Data> getGlobalVariableTable() {
        return GlobalVariableTable;
    }

    public ArrayList<IRLExpression> getGlobalAssignments() {
        return globalAssignments;
    }

    public void setGlobalAssignments(ArrayList<IRLExpression> globalAssignments) {
        this.globalAssignments = globalAssignments;
    }
}
