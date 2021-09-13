package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTFunction;
import de.parallelpatterndsl.patterndsl._ast.ASTTypeName;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl.coco.Helper.ReturnStatmentHelper;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, if a function needs a return statement.
 */
public class ReturnStatementExistsCoCo implements PatternDSLASTFunctionCoCo {

    @Override
    public void check(ASTFunction node) {
        if (!node.getPatternType().isPresentSerial()) {
            return;
        }
        if (node.getType() instanceof ASTTypeName) {
            if (((ASTTypeName) node.getType()).getName().equals("Void")) {
                return;
            }
        }
        ReturnStatmentHelper returnStatmentHelper = new ReturnStatmentHelper();
        if (!returnStatmentHelper.getExistsReturn(node)){
            Log.error(node.get_SourcePositionStart() + " No return found in: " + node.getName());
        }
    }



}
