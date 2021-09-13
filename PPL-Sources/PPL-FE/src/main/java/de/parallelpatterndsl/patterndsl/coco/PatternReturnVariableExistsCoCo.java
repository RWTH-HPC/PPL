package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._ast.ASTFunctionParameter;
import de.parallelpatterndsl.patterndsl._ast.ASTFunctionParameters;
import de.parallelpatterndsl.patterndsl._ast.ASTSerial;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if the return name is defined.
 */
public class PatternReturnVariableExistsCoCo implements PatternDSLASTFunctionCoCo {

    @Override
    public void check(ASTFunction node) {
        if (node.isPresentFunctionParameter() || node.getPatternType().isPresentSerial()) {
            return;
        }
        Log.error(node.get_SourcePositionStart() + " No return variable for this pattern defined: " + node.getName());
    }

}
