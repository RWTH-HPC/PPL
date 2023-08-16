package de.parallelpatterndsl.patterndsl.Preprocessing;

import de.parallelpatterndsl.patterndsl.abstractPatternTree.DataElements.Data;

import java.util.HashMap;
import java.util.Stack;


/**
 * Stores for each function scope, which data elements are replaced, by which inlined data elements.
 */
public class VariableReplacementStack {

    private static Stack<HashMap<Data, Data>> currentTableStack = new Stack<>();



    public static HashMap<Data, Data> getCurrentTable() {
        return currentTableStack.peek();
    }

    public static void removeLastTable() {
        currentTableStack.pop();
    }

    public static void addTable(HashMap<Data, Data> newTable) {
        currentTableStack.add(newTable);
    }

}
