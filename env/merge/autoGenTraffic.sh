#!/bin/bash
if [ -z "${SUMO_HOME}" ]; then
        export SUMO_HOME="/home/xdjf/miniconda3/envs/dreamer/lib/python3.12/site-packages/sumo"
    echo "SUMO_HOME was not set. Using default: ${SUMO_HOME}"
else
    echo "SUMO_HOME is already set to: ${SUMO_HOME}"
fi

python3 $SUMO_HOME/tools/createVehTypeDistribution.py car.config.txt --size 10000 --name "car"
python3 $SUMO_HOME/tools/createVehTypeDistribution.py bus.config.txt --size 1000 --name "bus"

python3 $SUMO_HOME/tools/randomTrips.py \
    -n highway.net.xml \
    -o car.trips.xml \
    -random \
    -p 0.27 \
    --random-depart \
    --fringe-factor 100000 \
    -L \
    --min-distance 10 \
    --max-distance 500000 \
    --end 754 \
    -r output.trips1.xml \
    --seed 70 \
    --validate \
    --trip-attributes "departLane=\"best\" departSpeed=\"5\"  " \
    --prefix car

python3 $SUMO_HOME/tools/randomTrips.py \
    -n highway.net.xml \
    -o bus.trips.xml \
    -p 3.14 \
    --fringe-factor 100000 \
    -L \
    --min-distance 10 \
    --max-distance 500000 \
    --end 754 \
    -r output.trips2.xml \
    --seed 30 \
    --validate \
    --trip-attributes "departLane=\"best\" departSpeed=\"5\" " \
    --prefix bus

perl -pi -e 's/(<trip id="([^"]+)")/\1 type="\2"/g' car.trips.xml
perl -pi -e 's/(<trip id="([^"]+)")/\1 type="\2"/g' bus.trips.xml
