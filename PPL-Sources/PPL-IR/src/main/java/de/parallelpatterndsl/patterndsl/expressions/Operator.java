package de.parallelpatterndsl.patterndsl.expressions;

import java.util.HashMap;

/**
 * A list of operators for operation expressions and their arities.
 */
public enum Operator {

    PLUS,
    MINUS,
    MULTIPLICATION,
    DIVIDE,
    MODULO,
    INCREMENT,
    DECREMENT,
    LESS,
    LESS_OR_EQUAL,
    EQUAL,
    NOT_EQUAL,
    GREATER,
    GREATER_OR_EQUAL,
    LOGICAL_NOT,
    LOGICAL_AND,
    LOGICAL_OR,
    BITWISE_NOT,
    BITWISE_OR,
    BITWISE_AND,
    LEFT_PARENTHESIS,
    RIGHT_PARENTHESIS,
    ASSIGNMENT,
    PLUS_ASSIGNMENT,
    MINUS_ASSIGNMENT,
    TIMES_ASSIGNMENT,
    LEFT_ARRAY_ACCESS,
    RIGHT_ARRAY_ACCESS,
    LEFT_CALL_PARENTHESIS,
    RIGHT_CALL_PARENTHESIS,
    COMMA,
    LEFT_ARRAY_DEFINITION,
    RIGHT_ARRAY_DEFINITION;


    private static final HashMap<Operator, Integer> arities;
    static {
        arities = new HashMap<>();
        arities.put(PLUS, 2);
        arities.put(MINUS, 2);
        arities.put(MULTIPLICATION, 2);
        arities.put(DIVIDE, 2);
        arities.put(MODULO, 2);
        arities.put(INCREMENT, 1);
        arities.put(DECREMENT, 1);
        arities.put(LESS, 2);
        arities.put(LESS_OR_EQUAL, 2);
        arities.put(EQUAL, 2);
        arities.put(NOT_EQUAL, 2);
        arities.put(GREATER, 2);
        arities.put(GREATER_OR_EQUAL, 2);
        arities.put(LOGICAL_NOT, 1);
        arities.put(LOGICAL_AND, 2);
        arities.put(LOGICAL_OR, 2);
        arities.put(BITWISE_NOT, 1);
        arities.put(BITWISE_OR, 2);
        arities.put(BITWISE_AND, 2);
        arities.put(LEFT_PARENTHESIS, 1);
        arities.put(RIGHT_PARENTHESIS, 1);
        arities.put(ASSIGNMENT, 2);
        arities.put(PLUS_ASSIGNMENT, 2);
        arities.put(MINUS_ASSIGNMENT, 2);
        arities.put(TIMES_ASSIGNMENT, 2);
        arities.put(LEFT_ARRAY_ACCESS,2);
        arities.put(RIGHT_ARRAY_ACCESS,1);
        arities.put(LEFT_CALL_PARENTHESIS,2);
        arities.put(RIGHT_CALL_PARENTHESIS,1);
        arities.put(COMMA,2);
        arities.put(LEFT_ARRAY_DEFINITION,1);
        arities.put(RIGHT_ARRAY_DEFINITION,1);


    }

    public static int arity(Operator input) {
        return arities.get(input);
    }


    private static final HashMap<Operator, Integer> operationCount;
    static {
        operationCount = new HashMap<>();
        operationCount.put(PLUS, 1);
        operationCount.put(MINUS, 1);
        operationCount.put(MULTIPLICATION, 1);
        operationCount.put(DIVIDE, 1);
        operationCount.put(MODULO, 1);
        operationCount.put(INCREMENT, 1);
        operationCount.put(DECREMENT, 1);
        operationCount.put(LESS, 1);
        operationCount.put(LESS_OR_EQUAL, 1);
        operationCount.put(EQUAL, 1);
        operationCount.put(NOT_EQUAL, 1);
        operationCount.put(GREATER, 1);
        operationCount.put(GREATER_OR_EQUAL, 1);
        operationCount.put(LOGICAL_NOT, 1);
        operationCount.put(LOGICAL_AND, 1);
        operationCount.put(LOGICAL_OR, 1);
        operationCount.put(BITWISE_NOT, 1);
        operationCount.put(BITWISE_OR, 1);
        operationCount.put(BITWISE_AND, 1);
        operationCount.put(LEFT_PARENTHESIS, 0);
        operationCount.put(RIGHT_PARENTHESIS, 0);
        operationCount.put(ASSIGNMENT, 0);
        operationCount.put(PLUS_ASSIGNMENT, 1);
        operationCount.put(MINUS_ASSIGNMENT, 1);
        operationCount.put(TIMES_ASSIGNMENT, 1);
        operationCount.put(LEFT_ARRAY_ACCESS,0);
        operationCount.put(RIGHT_ARRAY_ACCESS,0);
        operationCount.put(LEFT_CALL_PARENTHESIS,0);
        operationCount.put(RIGHT_CALL_PARENTHESIS,0);
        operationCount.put(COMMA,0);
        operationCount.put(LEFT_ARRAY_DEFINITION,0);
        operationCount.put(RIGHT_ARRAY_DEFINITION,0);


    }

    public static int getCount(Operator input) {
        return operationCount.get(input);
    }

}
