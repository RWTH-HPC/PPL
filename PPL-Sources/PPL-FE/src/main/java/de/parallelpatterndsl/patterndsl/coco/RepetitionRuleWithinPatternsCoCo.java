package de.parallelpatterndsl.patterndsl.coco;


import de.parallelpatterndsl.patterndsl._ast.ASTMinusExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTMultExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTNameExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTPlusExpression;
import de.parallelpatterndsl.patterndsl._ast.ASTExpression;

import de.monticore.literals.literals._ast.ASTIntLiteral;
import de.parallelpatterndsl.patterndsl._ast.*;
import de.parallelpatterndsl.patterndsl._cocos.PatternDSLASTFunctionCoCo;
import de.parallelpatterndsl.patterndsl._symboltable.FunctionParameterSymbol;
import de.parallelpatterndsl.patterndsl._visitor.PatternDSLVisitor;
import de.se_rwth.commons.logging.Log;


/**
 * Coco that checks, if the index accesses within the parallel patterns conform the repetition rule.
 */
public class RepetitionRuleWithinPatternsCoCo implements PatternDSLASTFunctionCoCo {
    @Override
    public void check(ASTFunction node) {
        if (node.getPatternType().isPresentSerial()) {
            return;
        }
        if (node.getPatternType().isPresentDynamicProgramming()) {
            DPHelper helper = new DPHelper();

            if (!helper.hasCorrectAccessPattern(node)) {
                Log.error(node.get_SourcePositionStart() + "Pattern access rule not correct in function: " + node.getName());
            }
            return;
        }
        Helper helper = new Helper();

        if (!helper.hasCorrectAccessPattern(node)) {
            Log.error(node.get_SourcePositionStart() + "Pattern access rule not correct in function: " + node.getName());
        }

    }

    private class DPHelper implements PatternDSLVisitor {
        private boolean isCorrect = true;

        private boolean accessesParameter = false;

        public DPHelper() {
        }

        public boolean hasCorrectAccessPattern(ASTFunction node) {
            node.accept(getRealThis());
            return isCorrect;
        }

        @Override
        public void visit(ASTIndexAccessExpression node) {
            if (node.getIndexAccess() instanceof ASTNameExpression) {
                if (node.getEnclosingScope().resolveMany(((ASTNameExpression) node.getIndexAccess()).getName(), FunctionParameterSymbol.KIND).size() == 0) {
                    accessesParameter = true;
                    return;
                }
            }
            accessesParameter = false;
        }

        @Override
        public void endVisit(ASTIndexAccessExpression node) {
            ASTExpression expression = node.getIndex();
            if (accessesParameter) {
                return;
            }

            if (expression instanceof ASTNameExpression) {
                if (!((ASTNameExpression) expression).getName().startsWith("INDEX1")) {
                    isCorrect = false;
                    Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                }
            } else if (expression instanceof ASTPlusExpression) {
                ASTExpression left = ((ASTPlusExpression) expression).getLeft();
                ASTExpression right = ((ASTPlusExpression) expression).getRight();

                if (right instanceof ASTLiteralExpression) {
                    if ( !(((ASTLiteralExpression) right).getLiteral() instanceof ASTIntLiteral)) {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }
                } else {
                    isCorrect = false;
                    Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                }

                if (left instanceof ASTNameExpression) {
                    if (!((ASTNameExpression) left).getName().startsWith("INDEX1")) {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }
                }
            } else if (expression instanceof ASTMinusExpression) {
                ASTExpression left = ((ASTMinusExpression) expression).getLeft();
                ASTExpression right = ((ASTMinusExpression) expression).getRight();

                if (right instanceof ASTLiteralExpression) {
                    if ( !(((ASTLiteralExpression) right).getLiteral() instanceof ASTIntLiteral)) {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }
                } else {
                    isCorrect = false;
                    Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                }

                if (left instanceof ASTNameExpression) {
                    if (!((ASTNameExpression) left).getName().startsWith("INDEX1")) {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }
                }

            } else if (expression instanceof ASTLiteralExpression) {
                if ( !(((ASTLiteralExpression) expression).getLiteral() instanceof ASTIntLiteral)) {
                    isCorrect = false;
                    Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                }
            } else {
                isCorrect = false;
                Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
            }
        }

        private PatternDSLVisitor realThis = this;

        @Override
        public PatternDSLVisitor getRealThis() {
            return realThis;
        }

        @Override
        public void setRealThis(PatternDSLVisitor realThis) {
            this.realThis = realThis;
        }

    }


    private class Helper implements PatternDSLVisitor {

        private boolean isCorrect = true;

        private boolean accessesParameter = false;

        public Helper() {
        }

        public boolean hasCorrectAccessPattern(ASTFunction node) {
            node.accept(getRealThis());
            return isCorrect;
        }

        @Override
        public void visit(ASTIndexAccessExpression node) {
            if (node.getIndexAccess() instanceof ASTNameExpression) {
                if (node.getEnclosingScope().resolveMany(((ASTNameExpression) node.getIndexAccess()).getName(), FunctionParameterSymbol.KIND).size() == 0) {
                    accessesParameter = true;
                    return;
                }
            }
            accessesParameter = false;
        }

        @Override
        public void endVisit(ASTIndexAccessExpression node) {
            ASTExpression expression = node.getIndex();
            if (accessesParameter) {
                return;
            }

            if (expression instanceof ASTNameExpression) {
                if (!((ASTNameExpression) expression).getName().startsWith("INDEX")) {
                    isCorrect = false;
                    Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                }
            } else if (expression instanceof ASTMultExpression) {
                ASTExpression left = ((ASTMultExpression) expression).getLeft();
                ASTExpression right = ((ASTMultExpression) expression).getRight();

                if (left instanceof ASTLiteralExpression) {
                    if ( !(((ASTLiteralExpression) left).getLiteral() instanceof ASTIntLiteral)) {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }
                } else {
                    isCorrect = false;
                    Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                }

                if (right instanceof ASTNameExpression) {
                    if (!((ASTNameExpression) right).getName().startsWith("INDEX")) {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }
                } else if (right instanceof ASTPlusExpression) {
                    ASTExpression plusLeft = ((ASTPlusExpression) right).getLeft();
                    ASTExpression plusRight = ((ASTPlusExpression) right).getRight();

                    if (plusLeft instanceof ASTNameExpression) {
                        if (!((ASTNameExpression) plusLeft).getName().startsWith("INDEX")) {
                            isCorrect = false;
                            Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                        }
                    }

                    if (plusRight instanceof ASTLiteralExpression) {
                        if ( !(((ASTLiteralExpression) plusRight).getLiteral() instanceof ASTIntLiteral)) {
                            isCorrect = false;
                            Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                        }
                    } else {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }

                } else if (right instanceof ASTMinusExpression) {
                    ASTExpression plusLeft = ((ASTMinusExpression) right).getLeft();
                    ASTExpression plusRight = ((ASTMinusExpression) right).getRight();

                    if (plusLeft instanceof ASTNameExpression) {
                        if (!((ASTNameExpression) plusLeft).getName().startsWith("INDEX")) {
                            isCorrect = false;
                            Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                        }
                    }

                    if (plusRight instanceof ASTLiteralExpression) {
                        if ( !(((ASTLiteralExpression) plusRight).getLiteral() instanceof ASTIntLiteral)) {
                            isCorrect = false;
                            Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                        }
                    } else {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }

                }

            } else if (expression instanceof ASTPlusExpression) {
                ASTExpression left = ((ASTPlusExpression) expression).getLeft();
                ASTExpression right = ((ASTPlusExpression) expression).getRight();

                if (right instanceof ASTLiteralExpression) {
                    if ( !(((ASTLiteralExpression) right).getLiteral() instanceof ASTIntLiteral)) {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }
                } else {
                    isCorrect = false;
                    Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                }

                if (left instanceof ASTNameExpression) {
                    if (!((ASTNameExpression) left).getName().startsWith("INDEX")) {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }
                } else if (left instanceof ASTMultExpression) {
                    ASTExpression multLeft = ((ASTMultExpression) left).getLeft();
                    ASTExpression multRight = ((ASTMultExpression) left).getRight();

                    if (multRight instanceof ASTNameExpression) {
                        if (!((ASTNameExpression) multRight).getName().startsWith("INDEX")) {
                            isCorrect = false;
                            Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                        }
                    }

                    if (multLeft instanceof ASTLiteralExpression) {
                        if ( !(((ASTLiteralExpression) multLeft).getLiteral() instanceof ASTIntLiteral)) {
                            isCorrect = false;
                            Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                        }
                    } else {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }

                }

            } else if (expression instanceof ASTMinusExpression) {
                ASTExpression left = ((ASTMinusExpression) expression).getLeft();
                ASTExpression right = ((ASTMinusExpression) expression).getRight();

                if (right instanceof ASTLiteralExpression) {
                    if ( !(((ASTLiteralExpression) right).getLiteral() instanceof ASTIntLiteral)) {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }
                } else {
                    isCorrect = false;
                    Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                }

                if (left instanceof ASTNameExpression) {
                    if (!((ASTNameExpression) left).getName().startsWith("INDEX")) {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }
                } else if (left instanceof ASTMultExpression) {
                    ASTExpression multLeft = ((ASTMultExpression) left).getLeft();
                    ASTExpression multRight = ((ASTMultExpression) left).getRight();

                    if (multRight instanceof ASTNameExpression) {
                        if (!((ASTNameExpression) multRight).getName().startsWith("INDEX")) {
                            isCorrect = false;
                            Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                        }
                    }

                    if (multLeft instanceof ASTLiteralExpression) {
                        if ( !(((ASTLiteralExpression) multLeft).getLiteral() instanceof ASTIntLiteral)) {
                            isCorrect = false;
                            Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                        }
                    } else {
                        isCorrect = false;
                        Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
                    }

                }

            } else {
                isCorrect = false;
                Log.error(node.get_SourcePositionStart() + " Pattern access rule not correct ");
            }
        }

        private PatternDSLVisitor realThis = this;

        @Override
        public PatternDSLVisitor getRealThis() {
            return realThis;
        }

        @Override
        public void setRealThis(PatternDSLVisitor realThis) {
            this.realThis = realThis;
        }
    }
}
