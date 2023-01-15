#!/bin/bash

port1_iface=$1
port2_iface=$2
port3_iface=$3


echo Hello, FABRIC. From node `hostname -s`
sudo apt-get update
sudo apt-get install -y docker.io

docker run -d -it --cap-add=NET_ADMIN --privileged --name fabric_p4 registry.ipv4.docker.com/pruth/fabric-images:0.0.2j
until [ "`docker inspect -f {{.State.Running}} fabric_p4`"=="true" ]; do
    sleep 0.1;
done;

NSPID=$(docker inspect --format='{{ .State.Pid }}' fabric_p4)
echo NSPID $NSPID

switch_interface_args=""
swtich_port=1
for iface in $@; do
  ip link set dev $iface promisc on
  ip link set $iface netns $NSPID
  docker exec fabric_p4 ip link set dev $iface up
  docker exec fabric_p4 ip link set dev $iface promisc on
  docker exec fabric_p4 sysctl net.ipv6.conf.${iface}.disable_ipv6=1

  switch_interface_args="${switch_interface_args} --interface ${swtich_port}@${iface}"
  swtich_port=$((swtich_port+1))
done

echo Starting switch
docker exec -w /root/tutorials/exercises/basic_tunnel fabric_p4 sh -c 'cp basic_tunnel.p4 basic_tunnel.working.p4'
docker exec -w /root/tutorials/exercises/basic_tunnel fabric_p4 sh -c 'cp solution/basic_tunnel.p4 basic_tunnel.p4'
docker exec -w /root/tutorials/exercises/basic_tunnel fabric_p4 sh -c 'p4c --p4runtime-files basic_tunnel.txt --target bmv2 --arch v1model basic_tunnel.p4'
docker exec -d -it fabric_p4 sh -c 'simple_switch --interface 1@'${port1_iface}' --interface 2@'${port2_iface}' --interface 3@'${port3_iface}' /root/tutorials/exercises/basic_tunnel/basic_tunnel.json'
echo 'simple_switch '${switch_interface_args}' /root/tutorials/exercises/basic_tunnel/basic_tunnel.json'

echo done!
