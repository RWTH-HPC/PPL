package de.parallelpatterndsl.patterndsl.coco;

import de.parallelpatterndsl.patterndsl._ast.ASTExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTCallExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTCallExpressionCoCo;
import de.parallelpatterndsl.patterndsl._ast.ASTListExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTLiteralExpression;
import de.se_rwth.commons.logging.Log;

/**
 * CoCo that checks, that for sequential functions only the return type is defined.
 */
public class Init_ListWithLiteralsCoCo implements PatternDSLASTCallExpressionCoCo {

    @Override
    public void check(ASTCallExpression node) {
        String name = ((ASTNameExpression)node.getCall()).getName();
        if (name.equals("init_List")) {
            if (node.getArguments().getExpressionList().size() == 0 || node.getArguments().getExpressionList().size() > 2) {
                Log.error(node.get_SourcePositionStart() + " Init_List must have 1 or 2 parameters!");
            } else {
                if (node.getArguments().getExpression(0) instanceof ASTListExpression) {
                    ASTListExpression list = (ASTListExpression) node.getArguments().getExpression(0);
                    for (ASTExpression exp : list.getExpressionList()) {
                        if (!(exp instanceof ASTLiteralExpression)) {
                            Log.error(node.get_SourcePositionStart() + " All parameters of the Init_List function must be literals! Variables are not allowed!");

                        }
                    }
                } else {
                    Log.error(node.get_SourcePositionStart() + " First parameter of Init_List must be of type list!");
                }
            }
        }
    }

}
