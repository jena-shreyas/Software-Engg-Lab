//Name : Shreyas Jena
//Roll : 20CS30049
//Assignment no: 2, part 2

/*
    This is PART-2 of the assignment.
    This program prints the k-closest nodes to a given node using the crow fly distance, k given as input.
*/

//include header files and definitions
#include <iostream>
#include "rapidxml_utils.hpp"
#include <cstring>
#include <cmath>
#include <queue>
#include <map>
#define lli long long int
#define RAD 6371
#define PI 3.14159

//include namespaces
using namespace std;
using namespace rapidxml;

//include typedefs
typedef struct {

    lli id;
    double lat;
    double lon;

}NODE;

typedef struct{

    lli id;
    vector<lli> nodes;
    
}WAY;

//function to parse node and store its details
NODE parseNode(xml_node<>* childNode){

    NODE node;

    node.id = stol(childNode->first_attribute("id")->value());
    node.lat = stod(childNode->first_attribute("lat")->value()); 
    node.lon = stod(childNode->first_attribute("lon")->value());

    return node;
}

//function to parse way and store its details
WAY parseWay(xml_node<>* childNode){

    WAY way;
    way.id = stol(childNode->first_attribute("id")->value());

    return way;
}

//function to print node details
void nodeDetails(NODE node){

    cout<<"Node id: "<<node.id<<endl;
    cout<<"Node latitude: "<<node.lat<<endl;
    cout<<"Node longitude: "<<node.lon<<endl;
}

//function to print distance between two points on Earth with given latitudes and longitudes using Haversine formula
double distance(NODE node1, NODE node2){

    double lat1 = node1.lat*(PI/180);
    double lat2 = node2.lat*(PI/180);
    double lon1 = node1.lon*(PI/180);
    double lon2 = node2.lon*(PI/180);

    double dLat = lat2-lat1;
    double dLon = lon2-lon1;

    return 2*RAD*asin(sqrt(pow(sin(dLat/2),2) + cos(lat1)*cos(lat2)*pow(sin(dLon/2),2)));
}

//function to parse osm file
void parseFile(map<lli,NODE> &hnode,map<lli,WAY> &hway,xml_node<> *rootNode){

    //here childNode pointer iterates over parsed xml file and accesses node/way elements one at a time
    for (xml_node<> *childNode = rootNode->first_node(); childNode; childNode = childNode->next_sibling()){

        //if current element is a node
        if (!strcmp(childNode->name(),"node")){
            
            xml_attribute<> *attr;
            NODE node = parseNode(childNode);                       //parse the node element    
            hnode.insert(pair<lli,NODE>(node.id,node));        //add node to map of nodes
        }
        
        //if current element is a way
        else if (!strcmp(childNode->name(),"way")) {

            WAY way = parseWay(childNode);                          //parse way element
            //here gchildNode pointer iterates over tags of each way one at a time
            for (xml_node<>* gchildNode = childNode->first_node("nd"); gchildNode; gchildNode = gchildNode->next_sibling("nd"))

                //add nodes present in way to its list of nodes
                way.nodes.push_back(stol(gchildNode->first_attribute("ref")->value()));
            
            hway.insert(pair<lli,WAY>(way.id,way));           //add way to map of ways
        }
    }
}

//the main function
int main(){

    //define variables of xml type to parse OSM file and store the parsed contents
    file<> xmlfile("map.osm");
    xml_document<> doc;
    doc.parse<0>(xmlfile.data());
    xml_node<>* rootNode = doc.first_node();

    //define maps to store hash map with key being node/way id and value being node/way object
    map<lli,NODE> hnode;
    map<lli,WAY> hway;

    parseFile(hnode,hway,rootNode);         //parse xml file
    
    cout<<"\n\n***PROGRAM TO FIND K-NEAREST NODES TO A GIVEN NODE IN A MAP***\n\n"<<endl;
    //Enter details of input node and value of k
    lli id;
    int k;
    cout<<"Enter id of node: ";
    cin>>id;
    auto itr = hnode.find(id);
    if (itr==hnode.end()){
        cout<<"The requested node was not found."<<endl;  
        exit(0);
    }

    cout<<"Enter value of k: ";
    cin>>k;

    //find input node details
    NODE findNode = itr->second;
    double lat = findNode.lat;
    double lon = findNode.lon;

    //define a min priority queue to store nodes in increasing order of distance from input node
    priority_queue<pair<double,long int>, vector<pair<double,long int>>, greater<pair<double,long int>>> pq;
    map<lli,NODE>::key_compare mycomp = hnode.key_comp();     //comparator object which compares ids of two nodes based on their order in node map
    map<lli,NODE>::iterator it = hnode.begin();               

    lli highest = hnode.rbegin()->first;
    double dist;

    //iterate over nodes in node map and add them to priority queue with priority=distance from input node
    do{

        NODE node = it->second;
        if (node.lat==lat && node.lon==lon)     continue;

        dist = distance(node,findNode);
        pq.push(make_pair(dist,node.id));

    }while(mycomp((*it++).first,highest));

    //print details of k-nearest nodes
    cout<<"\nDETAILS OF "<<k<<" NEAREST NODES:\n"<<endl;

    for (int i=1;i<=k;i++){

        pair<double,lli> pair = pq.top();
        NODE node = hnode[pair.second];
        nodeDetails(node);
        cout<<"Distance from node: "<<(pair.first)<<" km"<<endl;
        cout<<endl;
        pq.pop();
    }

    cout<<"Process of finding k-nearest nodes complete."<<endl;
    return 0;
}
//end of main



