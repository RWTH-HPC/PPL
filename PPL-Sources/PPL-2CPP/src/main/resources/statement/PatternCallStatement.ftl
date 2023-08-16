<#assign gh = glex.getGlobalVar("pfHelper")> <#-- GeneratorHelper -->
${signature("pattern")}
<#if gh.isMap(pattern)>
   ${includeArgs("statement/parallel/Map.ftl",pattern)}
</#if>
<#if gh.isReduction(pattern)>
   ${includeArgs("statement/parallel/Reduction.ftl",pattern)}
</#if>
<#if gh.isStencil(pattern)>
   ${includeArgs("statement/parallel/Stencil.ftl",pattern)}
</#if>
