// Name : Shreyas Jena
// Roll : 20CS30049
// Assignment no: 2, part 3

/*
    This is PART-3 of the assignment.
    This program prints the shortest path between two node elements, through the way elements using Dijkstra's algorithm
*/

// include header files and definitions
#include <iostream>
#include "rapidxml_utils.hpp"
#include <cstring>
#include <cmath>
#include <queue>
#include <map>
#include <set>
#define lli long long int
#define RAD 6371
#define PI 3.14159

// include namespaces
using namespace std;
using namespace rapidxml;

// include typedefs
typedef struct node
{

    lli id;
    double lat;
    double lon;
    double cost = 0.0;
    struct node *parent;

} NODE;

typedef struct
{

    lli id;
    vector<lli> nodes;

} WAY;

typedef priority_queue<pair<double, lli>, vector<pair<double, lli>>, greater<pair<double, lli>>> pq_min;

// function to print distance between two points on Earth with given latitudes and longitudes using Haversine formula
double distance(NODE *nodep1, NODE *nodep2)
{

    double lat1 = nodep1->lat * (PI / 180);
    double lat2 = nodep2->lat * (PI / 180);
    double lon1 = nodep1->lon * (PI / 180);
    double lon2 = nodep2->lon * (PI / 180);

    double dLat = lat2 - lat1;
    double dLon = lon2 - lon1;

    return 2 * RAD * asin(sqrt(pow(sin(dLat / 2), 2) + cos(lat1) * cos(lat2) * pow(sin(dLon / 2), 2)));
}

// function to parse node and store its details
NODE parseNode(xml_node<> *childNode)
{

    NODE node;

    node.id = stol(childNode->first_attribute("id")->value());
    node.lat = stod(childNode->first_attribute("lat")->value());
    node.lon = stod(childNode->first_attribute("lon")->value());
    node.parent = NULL;
    return node;
}

// function to parse way and store its details
WAY parseWay(xml_node<> *childNode)
{

    WAY way;
    way.id = stol(childNode->first_attribute("id")->value());

    return way;
}

// function to print the sequence of nodes in path from source to destination
void findPath(NODE *endNode, int &path_nodes)
{

    if (endNode == NULL)
        return;

    findPath(endNode->parent, ++path_nodes);
    cout << endNode->id << endl;
}

// function to parse osm file
void parseFile(map<lli, NODE> &hnode, map<lli, WAY> &hway, map<lli, vector<lli>> &adjList, xml_node<> *rootNode)
{

    // parsing the xml file
    for (xml_node<> *childNode = rootNode->first_node(); childNode; childNode = childNode->next_sibling())
    {

        if (!strcmp(childNode->name(), "node"))
        {

            xml_attribute<> *attr;
            NODE node = parseNode(childNode);
            hnode.insert(pair<lli, NODE>(node.id, node));
        }

        else if (!strcmp(childNode->name(), "way"))
        {

            int i;
            lli id;
            WAY way = parseWay(childNode);
            for (xml_node<> *gchildNode = childNode->first_node("nd"); gchildNode; gchildNode = gchildNode->next_sibling("nd"))
            {

                // add nodes present in way to its nodes list
                id = stol(gchildNode->first_attribute("ref")->value());
                way.nodes.push_back(id);
            }

            // add each node in given way with its neighbours in adjacency list
            for (i = 0; i < way.nodes.size(); i++)
            {

                auto itr = adjList.find(way.nodes[i]);
                if (itr != adjList.end())
                { // if node already present in adjacency list

                    if (i == 0)
                        itr->second.push_back(way.nodes[1]);

                    else if (i == way.nodes.size() - 1)
                        itr->second.push_back(way.nodes[way.nodes.size() - 2]);

                    else
                    {

                        itr->second.push_back(way.nodes[i - 1]);
                        itr->second.push_back(way.nodes[i + 1]);
                    }
                }
                else
                { // node not present in adjacency list

                    vector<lli> v;
                    if (i == 0)
                        v.push_back(way.nodes[1]);

                    else if (i == way.nodes.size() - 1)
                        v.push_back(way.nodes[way.nodes.size() - 2]);

                    else
                    {

                        v.push_back(way.nodes[i - 1]);
                        v.push_back(way.nodes[i + 1]);
                    }

                    adjList.insert(pair<lli, vector<lli>>(way.nodes[i], v)); // add node with its neighbours to adjacency list
                }
            }
            hway.insert(pair<lli, WAY>(way.id, way));
        }
    }
}

// function to find shortest path usind Dijkstra algorithm
void dijkstra(lli snid, lli dnid, map<lli, NODE> &hnode, map<lli, WAY> &hway, map<lli, vector<lli>> &adjList, set<lli> &visited, pq_min &visiting)
{

    // for each iteration, pop out the node with min cost and update its neighbours' costs by edge relaxation
    while (!visiting.empty())
    {

        lli currnid = visiting.top().second;
        visiting.pop();

        // if current node is destination node
        if (currnid == dnid)
        {
            break;
        }

        auto it = adjList.find(currnid);
        // if current node has no neighbours, i.e. it doesn't lie on a path, no path can be found, hence break out of loop
        if (it == adjList.end())
            break;

        vector<lli> v;
        v = it->second; // neighbors' list of current node

        // iterate over each neighbor, update its cost if needed and put it in visiting queue
        for (int i = 0; i < v.size(); i++)
        {

            lli nb = v[i];
            if (visited.find(nb) != visited.end()) // if neighbor is already visited, skip
                continue;

            NODE *currNode = &(hnode[currnid]);
            NODE *nbNode = &(hnode[nb]);

            // edge relaxation step
            if (distance(currNode, nbNode) + currNode->cost < nbNode->cost)
            {

                nbNode->cost = distance(currNode, nbNode) + currNode->cost; // update neighbor's cost
                nbNode->parent = currNode;                                  // update its parent to current node
                visiting.push(pair<double, lli>(nbNode->cost, nb));         // add neighbor to visiting list
                (hnode[nb]) = *nbNode;                                      // update changes in node map
            }
        }

        visited.insert(currnid); // add current node to visited nodes set
    }
}

// the main function
int main()
{

    // define variables of xml type to parse OSM file and store the parsed contents
    file<> xmlfile("map.osm");
    xml_document<> doc;
    doc.parse<0>(xmlfile.data());
    xml_node<> *rootNode = doc.first_node();

    // define maps to store hash map with key being node/way id and value being node/way object
    map<lli, NODE> hnode;
    map<lli, WAY> hway;
    map<lli, vector<lli>> adjList; // define an adjacency list to store neighbours of each node

    parseFile(hnode, hway, adjList, rootNode); // parse the osm file

    cout<<"\n\n***PROGRAM TO FIND SHORTEST PATH BETWEEN TWO POINTS ON A MAP***\n\n"<<endl;
    // enter source and node id
    lli snid, dnid;
    cout << "Enter source node id: ";
    cin >> snid;

    auto it = hnode.find(snid);
    if (it==hnode.end()){
        cout<<"The source node was not found."<<endl;  
        exit(0);
    }

    cout << "Enter destination node id: ";
    cin >> dnid;

    it = hnode.find(dnid);
    if (it==hnode.end()){
        cout<<"The destination node was not found."<<endl;  
        exit(0);
    }

    cout << "\nFinding out the shortest path between " << snid << " and " << dnid << " ...\n"
         << endl;
    // iterate over nodes in node list and update their costs to max value if not source node, else 0
    map<lli, NODE>::iterator itr;
    for (itr = hnode.begin(); itr != hnode.end(); itr++)
    {

        if (itr->first == snid)
            (itr->second).cost = 0.0;

        else
            (itr->second).cost = __DBL_MAX__;
    }

    set<lli> visited; // define set storing the visited nodes
    // define a min priority queue storing the nodes currently active in order of increasing costs
    pq_min visiting;
    visiting.push(pair<double, lli>(0.0, snid));

    dijkstra(snid, dnid, hnode, hway, adjList, visited, visiting); // apply Dijkstra algorithm to find the shortest path

    NODE endNode;
    int path_nodes = 0;
    endNode = hnode[dnid]; // get destination node details

    // if destination node has no parent, path doesn't exist
    if (endNode.parent == NULL)
        cout << "Sorry, no path could be found between " << snid << " and " << dnid << "." << endl;
    else
    {
        // print path details
        cout << "Shortest path between " << snid << " and " << dnid << " found!" << endl;
        cout<<"\nPATH DETAILS:\n"<<endl;
        cout<<"Source id: "<<snid<<endl;
        cout<<"Destination id: "<<dnid<<endl;
        cout<<"\nIntermediate nodes found in shortest path are:\n"<<endl;
        findPath(&endNode, path_nodes);
        cout << "\nTotal intermediate nodes in shortest path: " << path_nodes << endl;
        cout << "Length of shortest path: " << endNode.cost << " km" << endl;
        cout<<"\nPath finding process complete."<<endl;
    }

    return 0;
}
// end of main