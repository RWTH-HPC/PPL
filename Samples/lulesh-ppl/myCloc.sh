#!/bin/bash
echo "myCloc: for correct results execute this in LULESH/ppl"
DATE=$(date +%d-%m-%Y_%H:%M)
cloc --by-file ./ --report-file=cloc-reports/cloc-report_$DATE  --force-lang="c",par --autoconf --exclude-dir=clusters,mappings,out,cloc-reports