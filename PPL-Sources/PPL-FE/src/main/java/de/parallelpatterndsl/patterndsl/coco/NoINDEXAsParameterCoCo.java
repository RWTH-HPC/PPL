package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._ast.ASTFunctionParameter;
import de.parallelpatterndsl.patterndsl._ast.ASTTypeName;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl.coco.Helper.ReturnStatmentHelper;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if a function needs a return statement.
 */
public class NoINDEXAsParameterCoCo implements PatternDSLASTFunctionCoCo {

    @Override
    public void check(ASTFunction node) {
        for (ASTFunctionParameter parameter: node.getFunctionParameters().getFunctionParameterList() ) {
            if (parameter.getName().startsWith("INDEX")) {
                Log.error(parameter.get_SourcePositionStart() + " Do not use INDEX variables as a parameter, they are a keyword in parallel patterns. Please rename this variable: " + parameter.getName());
            }
        }
    }



}
