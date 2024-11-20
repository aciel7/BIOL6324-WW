cd ./auxiliary_files/
wget "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?mode=raw&is_datatable=true&acc=GPL570&id=55999&db=GeoDb_blob143" -O GPL570_probeset.tsv
sed -i "/#/d" GPL570_probeset.tsv
wget "https://www.ncbi.nlm.nih.gov/geo/browse/?view=samples&series=203024&zsort=date&mode=csv&page=undefined&display=5000" -O samples.csv

