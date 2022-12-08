//Name : Shreyas Jena
//Roll : 20CS30049
//Assignment no: 2, part 1

/*
    This is PART-1 of the assignment.
    This program parses the given OSM file for Kharagpur area and extracts node and way elements along with their attributes.
    It allows a case-insensitive search for a string among those nodes which have tags with the name attribute.
*/

//include header files and definitions
#include <iostream>
#include "rapidxml_utils.hpp"
#include <cstring>
#include <algorithm>
#include <map>
#define lli long long int

//include namespaces
using namespace std;
using namespace rapidxml;

//include typedefs
typedef struct node{

    lli id;
    double lat;
    double lon;

}NODE;

typedef struct{

    lli id;
    vector<lli> nodes;
    
}WAY;

//function to find (case-insensitive) whether input string present in given string or not
bool findCaseInsensitive(string name, string str){

    transform(name.begin(),name.end(),name.begin(),::tolower);
    transform(str.begin(),str.end(),str.begin(),::tolower);
    return (name.find(str)!=string::npos);
}

//function to parse node and store its details
NODE parseNode(xml_node<>* childNode){

    NODE node;

    node.id = stol(childNode->first_attribute("id")->value());
    node.lat = stod(childNode->first_attribute("lat")->value()); 
    node.lon = stod(childNode->first_attribute("lon")->value());

    return node;
}

//function to print node details
void nodeDetails(NODE node){

    cout<<"Node id: "<<node.id<<endl;
    cout<<"Node latitude: "<<node.lat<<endl;
    cout<<"Node longitude: "<<node.lon<<endl;
}

//function to parse way and store its details
WAY parseWay(xml_node<>* childNode){

    WAY way;
    way.id = stol(childNode->first_attribute("id")->value());

    return way;
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
    int count_nodes=0;
    int count_ways=0;
    int search_results=0;       //counts the total no. of matching search results found

    cout<<"\n\n***PROGRAM TO FIND OCCURENCES OF A STRING AMONG NODES IN THE MAP***\n\n"<<endl;
    string str;                 //enter input string to be searched
    cout<<"Enter string to be searched: ";
    getline(cin,str);
    cout<<"\nRESULTS OF STRING SEARCH: \n"<<endl;
    
    //here childNode pointer iterates over parsed xml file and accesses node/way elements one at a time
    for (xml_node<> *childNode = rootNode->first_node(); childNode; childNode = childNode->next_sibling()){

        //if current element is a node
        if (!strcmp(childNode->name(),"node")){         
            
            count_nodes++;                      //increment no. of nodes
            NODE node = parseNode(childNode);   //parse the node element

            //here gchildNode pointer iterates over tags of each node one at a time
            for (xml_node<> *gchildNode = childNode->first_node(); gchildNode; gchildNode = gchildNode->next_sibling()){

                //if current element is a tag with attribute k="name",search for input string in value of its attribute v
                if (!strcmp(gchildNode->first_attribute("k")->value(),"name")){

                    string name = gchildNode->first_attribute("v")->value();        //stores the string to be searched
                    
                    //if input string found
                    if (findCaseInsensitive(name,str)){                             
                        
                        search_results++;
                        cout<<"\nSearch result: "<<name<<endl;
                        nodeDetails(node);             //print details of corresp node
                    }

                }
            }

            hnode.insert(pair<lli,NODE>(node.id,node));                             //add node to map of nodes
        }
        
        //if current element is a way
        else if (!strcmp(childNode->name(),"way")) {

            count_ways++;                                                           //increment no. of nodes
            WAY way=parseWay(childNode);                                            //parse way element

            //here gchildNode pointer iterates over tags of each way one at a time
            for (xml_node<>* gchildNode = childNode->first_node("nd"); gchildNode; gchildNode = gchildNode->next_sibling("nd"))

                //add nodes present in way to its list of nodes
                way.nodes.push_back(stol(gchildNode->first_attribute("ref")->value()));
            
            hway.insert(pair<lli,WAY>(way.id,way));                                 //add way to map of ways
        }
    }
    
    if (!search_results)    
        cout<<"\nGiven search string was not found."<<endl;
    else 
        cout<<"\n\nTotal matching results for "<<str<<": "<<search_results<<endl;

    cout<<"Total no. of nodes: "<<count_nodes<<endl;
    cout<<"Total no. of ways: "<<count_ways<<endl; 
    cout<<"\nProcess of parsing osm file and string matching complete."<<endl;
    return 0;
}
//end of main



