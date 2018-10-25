#! /bin/bash
project_name=$1

if [ -z ${project_name} ]; then
   echo "project name is empty"
   exit 1
fi


#cd ./TrendFormula

# generate a dummy log file to distinguish each team
touch ${project_name}.dummy

sudo docker build -t formula-trend .

sudo docker tag formula-trend ai.registry.trendmicro.com/${project_name}/formula-trend:rank
sudo docker tag formula-trend ai.registry.trendmicro.com/${project_name}/formula-trend:rank.1
sudo docker tag formula-trend ai.registry.trendmicro.com/${project_name}/formula-trend:rank.2
sudo docker tag formula-trend ai.registry.trendmicro.com/${project_name}/formula-trend:rank.3

sudo docker login ai.registry.trendmicro.com

sudo docker push ai.registry.trendmicro.com/${project_name}/formula-trend:rank
sudo docker push ai.registry.trendmicro.com/${project_name}/formula-trend:rank.1
sudo docker push ai.registry.trendmicro.com/${project_name}/formula-trend:rank.2
sudo docker push ai.registry.trendmicro.com/${project_name}/formula-trend:rank.3
