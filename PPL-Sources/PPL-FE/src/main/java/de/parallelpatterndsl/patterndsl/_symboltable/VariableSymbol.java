package de.parallelpatterndsl.patterndsl._symboltable;

import de.parallelpatterndsl.patterndsl._ast.ASTType;

import java.util.ArrayList;

/**
 * A symbol to store name and type of a Variable.
 * For variables of type list the shape is also stored.
 */
public class VariableSymbol extends VariableSymbolTOP {

    /**
     * Definition of the data type the variable has.
     */
    private ASTType type;

    /**
     * Definition of the shape the variable has. e.g. [2,3] a matrix with 2 rows and 3 columns.
     */
    private ArrayList<Integer> shape;

    /**
     * True, iff it is an array and is stored on the stack (Initialization with the List Expression).
     */
    private boolean arrayOnStack;


    public VariableSymbol(String name) {
        super(name);
    }

    public VariableSymbol(String name, ASTType type) {
        super(name);
        this.type = type;
        shape = new ArrayList<>();
    }

    public VariableSymbol(String name, ASTType type, boolean arrayOnStack) {
        super(name);
        this.type = type;
        shape = new ArrayList<>();
        this.arrayOnStack = arrayOnStack;
    }

    public ASTType getType() { return this.type;}

    public ArrayList<Integer> getShape() {
        return shape;
    }

    public void setShape(ArrayList<Integer> shape) {
        this.shape = shape;
    }

    public boolean isArrayOnStack() {
        return arrayOnStack;
    }
}
