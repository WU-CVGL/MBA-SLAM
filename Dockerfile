FROM mba-slam-base

ADD external/MBA-VO /opt/MBA-VO
RUN cd /opt/MBA-VO &&\
    cmake -GNinja -B build &&\
    cmake --build build -t install &&\
    cd python &&\
    pip install . &&\
    rm -rf ../build &&\
    rm -rf /tmp/*

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt &&\
    pip install pypose --no-deps &&\
    rm -rf /tmp/*
