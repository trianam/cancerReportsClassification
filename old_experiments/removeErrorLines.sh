#!/bin/sh

cd corpusLSTM_ICDO3-separated

sed -i.bak -e '35971d;36362d;51947d;89457d;90221d' text.txt
sed -i.bak -e '35971d;36362d;51947d;89457d;90221d' textNotizie.txt
sed -i.bak -e '35971d;36362d;51947d;89457d;90221d' textDiagnosi.txt
sed -i.bak -e '35971d;36362d;51947d;89457d;90221d' textMacroscopia.txt
sed -i.bak -e '35971d;36362d;51947d;89457d;90221d' morfoICDO3.txt
sed -i.bak -e '35971d;36362d;51947d;89457d;90221d' morfoICDO3clean.txt
sed -i.bak -e '35971d;36362d;51947d;89457d;90221d' sedeICDO3.txt
sed -i.bak -e '35971d;36362d;51947d;89457d;90221d' sedeICDO3clean.txt
#sed -i.bak -e '35971d;36362d;51947d;89457d;90221d' operatoreIns.txt
#sed -i.bak -e '35971d;36362d;51947d;89457d;90221d' operatoreAgg.txt

