 <#compress>
${signature("func")}
<#assign gh = glex.getGlobalVar("pfHelper")> <#-- GeneratorHelper -->
<#assign BlockElements=func.getBlockElementList()>
<#list BlockElements as BlockElement>
    <#if BlockElement.getExpressionOpt().isPresent()>
        ${gh.printExpression(BlockElement.getExpression())};
    </#if>
    <#if BlockElement.getVariableOpt().isPresent()>
        ${includeArgs("definition/Variable.ftl", BlockElement.getVariable(), "")}
    </#if>
    <#if BlockElement.getStatementOpt().isPresent()>
        <#assign Statement=BlockElement.getStatement()>
        <#if gh.isForStatement(Statement)>
            ${includeArgs("statement/ForLoop.ftl",Statement)}
        </#if>
        <#if gh.isIfStatement(Statement)>
            ${includeArgs("statement/IfStatement.ftl",Statement)}
        </#if>
        <#if gh.isReturnStatement(Statement)>
            ${includeArgs("statement/ReturnStatement.ftl",Statement)}
        </#if>
        <#if gh.isWhileStatement(Statement)>
            ${includeArgs("statement/WhileLoop.ftl",Statement)}
        </#if>
        <#if gh.isPatternCallStatement(Statement)>
            ${includeArgs("statement/PatternCallStatement.ftl",Statement)}
        </#if>
    </#if>
</#list>
</#compress>
