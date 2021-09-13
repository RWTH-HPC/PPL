package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._ast.ASTListType;
import de.parallelpatterndsl.patterndsl._ast.ASTType;
import de.parallelpatterndsl.patterndsl._ast.ASTTypeName;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl.coco.Helper.ReturnStatmentHelper;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if a function needs a return statement.
 */
public class DynamicProgrammingSyntaxCoCo implements PatternDSLASTFunctionCoCo {

    @Override
    public void check(ASTFunction node) {
        if (!node.getPatternType().isPresentDynamicProgramming()) {
            return;
        }
        if (!(node.getFunctionParameters().getFunctionParameterList().size() == 1)) {
            Log.error(node.get_SourcePositionStart() + " Dynamic programming functions only accept a single argument: " + node.getName());
        }
        ASTType type = node.getFunctionParameters().getFunctionParameterList().get(0).getType();
        if (type instanceof ASTListType) {
            if (((ASTListType) type).getType() instanceof ASTListType) {
                Log.error(node.get_SourcePositionStart() + " Dynamic programming functions only accept one dimensional lists or singular values: " + node.getName());
            }
        }
        if (!type.deepEquals(node.getFunctionParameter().getType())) {
            Log.error(node.get_SourcePositionStart() + " Input and output values must be of the same type: " + node.getName());
        }
    }



}
