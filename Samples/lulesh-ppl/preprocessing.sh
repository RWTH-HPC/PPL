#!/bin/bash
# replace variables in array defintions by numbers ----------------------------
# create copy of template
cat lulesh.par.in > lulesh.par
cat luleshSeq.par.in > luleshSeq.par

# handle profiling lines
PROFILING_On=false
if [ $PROFILING_On = true ]; then
    profiling=" "
    sed -i "s/\$PROFILING/$profiling/g" lulesh.par
else
    profiling="\/\/"
    sed -i "s/\$PROFILING/$profiling/g" lulesh.par
fi

maxIterations=$((1))

# moved from domain constructor (lulesh-init.cc)
# normally assigned by cmdOpts[nx]

nx=$((240))

edgeElems=$nx 
edgeNodes=$(($edgeElems + 1))
domain_numElem=$(($edgeElems * $edgeElems * $edgeElems))
domain_numNode=$(($edgeNodes * $edgeNodes * $edgeNodes))

#moved from CalcQForElems()
# var Int32 allElem = numElem +  /* local elem */
#           2 * domain_sizeX * domain_sizeY + /* plane ghosts */
#            2 * domain_sizeX * domain_sizeZ + /* row ghosts */
#            2 * domain_sizeY * domain_sizeZ /* col ghosts */
domain_allElem=$(($domain_numElem + 2 * $edgeElems * $edgeElems + 2 * $edgeElems * $edgeElems))

sed -i "s/\$maxIterations/$maxIterations/g" lulesh.par
sed -i "s/\$nx/$nx/g" lulesh.par luleshSeq.par
sed -i "s/\$edgeElems/$edgeElems/g" lulesh.par luleshSeq.par
sed -i "s/\$edgeNodes/$edgeNodes/g" lulesh.par luleshSeq.par
sed -i "s/\$domain_numElem/$domain_numElem/g" lulesh.par luleshSeq.par
sed -i "s/\$domain_numNode/$domain_numNode/g" lulesh.par luleshSeq.par

# java --add-opens java.base/java.lang=ALL-UNNAMED -jar ./PP_OPT.jar --input=lulesh.par --network=clusters/cluster_c18g.json
