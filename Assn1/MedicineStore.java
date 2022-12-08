//Name : Shreyas Jena
//Roll no : 20CS30049
//Software Engineering Lab - Assignment 1

import java.util.*;

//class to define a medicine store
public class MedicineStore{

    HashMap<String,Manufacturer> hman;
    HashMap<String,Customer> hcust;
    HashMap<String,Product> hprod;
    HashMap<String,Shop> hshop;
    HashMap<String,Warehouse> hware;
    HashMap<String,DeliveryAgent> hdelv;
    List<Order> orderList ;

    //initialize class variables using constructor
    public MedicineStore(){

        this.hman = new HashMap<>();
        this.hcust = new HashMap<>();
        this.hprod = new HashMap<>();;
        this.hshop = new HashMap<>();
        this.hware = new HashMap<>();
        this.hdelv = new HashMap<>();
        this.orderList = new ArrayList<>();
    }

//function to display menu for showing different functionalities of online medicine shop
    public void displayMenu(){

        System.out.println("\n**********FUNCTIONALITY MENU**********\n");
        System.out.println("\nEnter the appropriate number for any required functionality...\n");
        System.out.println("1.      Create, delete and print entities of each type");
        System.out.println("2.      Add a product to manufacturer");
        System.out.println("3.      Add a certain number of copies of a product to a shop");
        System.out.println("4.      Add an order of a product to a customer");
        System.out.println("5.      Process an order and assign an appropriate delivery agent");
        System.out.println("6.      List all the purchases made by a customer");
        System.out.println("7.      List the inventory of a shop (Products and counts)");
        System.out.println("8.      List the products made by a manufacturer\n");
        System.out.println("Press 0 to exit\n");
    }

//function to create entities
    public void createEntities(Scanner in){

        System.out.println("\n\n**********MENU FOR CREATING OBJECTS**********\n");
        System.out.println("a.      Create a Manufacturer");
        System.out.println("b.      Create a Customer");
        System.out.println("c.      Create a Product");
        System.out.println("d.      Create a Shop");
        System.out.println("e.      Create a Warehouse");
        System.out.println("f.      Create a Delivery Agent\n");
        System.out.println("Press q to quit creating entities\n");
        System.out.print("Please enter your choice: ");
        char choice=in.next().charAt(0);
        in.nextLine();

        while (choice!='q'){

            switch(choice){

                case 'a' :  System.out.println("\nPlease enter manufacturer details...\n");
                            Manufacturer man = new Manufacturer(in);
                            System.out.println("\nMANUFACTURER DETAILS:-\n");
                            man.manufacturerDetails();
                            this.hman.put(man.name,man);        //put manufacturer in manufacturer's list
                            break;

                case 'b' :  System.out.println("\nPlease enter customer details...\n");
                            Customer cust = new Customer(in);
                            System.out.println("\nCUSTOMER DETAILS:-\n");
                            cust.customerDetails();
                            this.hcust.put(cust.name,cust);     //put customer in customer's list
                            break;

                case 'c' :  System.out.println("\nPlease enter product details...\n");
                            Product p = new Product(in);  
                            System.out.println("\nPRODUCT DETAILS:-\n");
                            p.productDetails();
                            this.hprod.put(p.name,p);           //put product in product's list
                            break;

                case 'd' :  System.out.println("\nPlease enter shop details...\n");
                            Shop shop = new Shop(in);
                            System.out.println("\nSHOP DETAILS:-\n");
                            shop.shopDetails();
                            this.hshop.put(shop.name,shop);     //put shop in shop's list
                            break;

                case 'e' :  System.out.println("\nPlease enter warehouse details...\n");
                            Warehouse ware = new Warehouse(in);
                            System.out.println("\nWAREHOUSE DETAILS:-\n");
                            ware.warehouseDetails();
                            this.hware.put(ware.name,ware);     //put warehouse in warehouse's list
                            break;   
                            
                case 'f'  : System.out.println("\nPlease enter delivery agent details...\n");
                            DeliveryAgent delv = new DeliveryAgent(in);
                            System.out.println("\nDELIVERY AGENT DETAILS:-\n");
                            delv.deliveryAgentDetails();
                            this.hdelv.put(delv.name,delv);     //put delivery agent in delivery agent's list
                            break;

                default   : System.out.println("You entered an invalid option, please try again!");
            }

            System.out.print("\n\nPlease enter your choice: ");
            choice = in.next().charAt(0);
            in.nextLine();
        }

        System.out.println("\n\nProcess of creating objects complete.");
    }

//function to print entities
    public void printEntities(Scanner in){

        System.out.println("\nSelect an option to print an entity of any type...\n");
        System.out.println("\n**********MENU FOR PRINTING OBJECTS**********\n");
        System.out.println("a.      Print a Manufacturer");
        System.out.println("b.      Print a Customer");
        System.out.println("c.      Print a Product");
        System.out.println("d.      Print a Shop");
        System.out.println("e.      Print a Warehouse");
        System.out.println("f.      Print a Delivery Agent\n");
        System.out.println("Press q to quit printing entities\n");
        System.out.print("Please enter your choice: ");
        char choice=in.next().charAt(0);
        in.nextLine();
        String name;

        while (choice!='q'){

            switch(choice){

                case 'a' :  if (this.hman.size()==0)
                                    System.out.println("\nNo manufacturers left.");

                            else {  

                                    System.out.println("\nThe available manufacturers are: "+this.hman.keySet());
                                    System.out.print("Now enter name of manufacturer: ");
                                    name = in.nextLine();
                                    this.hman.get(name).manufacturerDetails();  //print manufacturer details
                            }

                            break;

                case 'b' :  if (this.hcust.size()==0)
                                    System.out.println("\nNo customers left.");
                                
                            else {
                                
                                    System.out.println("\nThe available customers are: "+this.hcust.keySet());
                                    System.out.print("Now enter name of customer: ");
                                    name = in.nextLine();
                                    this.hcust.get(name).customerDetails();     //print customer details
                            }

                            break;

                case 'c' :  if (this.hprod.size()==0)
                                    System.out.println("\nNo products left.");  

                            else {

                                    System.out.println("\nThe available products are: "+this.hprod.keySet());
                                    System.out.print("Now enter name of product: ");
                                    name = in.nextLine();
                                    this.hprod.get(name).productDetails();      //print product details
                                }

                            break;

                case 'd' :  if (this.hshop.size()==0)
                                    System.out.println("\nNo shops left."); 
                
                            else {

                                    System.out.println("\nThe available shops are: "+this.hshop.keySet());
                                    System.out.print("Now enter name of shop: ");
                                    name = in.nextLine();
                                    this.hshop.get(name).shopDetails();         //print shop details
                            }

                            break;

                case 'e' :  if (this.hware.size()==0)
                                    System.out.println("\nNo warehouses left.");
                
                            else {

                                    System.out.println("\nThe available warehouses are: "+this.hware.keySet());
                                    System.out.print("Now enter name of warehouse: ");
                                    name = in.nextLine();
                                    this.hware.get(name).warehouseDetails();    //print warehouse details
                            }

                            break;   
                            
                case 'f'  : if (this.hdelv.size()==0)
                                    System.out.println("\nNo delivery agents left.");

                            else {

                                    System.out.println("\nThe available delivery agents are: "+this.hdelv.keySet());
                                    System.out.print("Now enter name of delivery agent: ");
                                    name = in.nextLine();
                                    this.hdelv.get(name).deliveryAgentDetails(); //print delivery agent details
                            }

                            break;

                default   : System.out.println("You entered an invalid option, please try again!");
            }

            System.out.print("\n\nPlease enter your choice: ");
            choice = in.next().charAt(0);
            in.nextLine();
        }

        System.out.println("\n\nProcess of printing objects complete");
    }

//function to delete entities
    public void deleteEntities(Scanner in){

        System.out.println("\nSelect an option to delete an entity of any type...\n");
        System.out.println("\n**********MENU FOR DELETING OBJECTS**********\n");
        System.out.println("a.      Delete a Manufacturer");
        System.out.println("b.      Delete a Customer");
        System.out.println("c.      Delete a Product");
        System.out.println("d.      Delete a Shop");
        System.out.println("e.      Delete a Warehouse");
        System.out.println("f.      Delete a Delivery Agent\n");
        System.out.println("Press q to quit deleting entities\n");
        System.out.print("Please enter your choice: ");
        char choice=in.next().charAt(0);
        in.nextLine();
        String name;

        while (choice!='q'){

            switch(choice){

                case 'a' :  if (this.hman.size()==0)
                                    System.out.println("\nNo manufacturers left.");

                            else {  

                                    System.out.println("\nThe available manufacturers are: "+this.hman.keySet());
                                    System.out.print("Now enter name of manufacturer to be deleted: ");
                                    name = in.nextLine();

                                    //remove selected manufacturer from manufacturer's list
                                    Manufacturer man = this.hman.remove(name);
                                    if (man == null)
                                        System.out.println("This manufacturer does not exist");

                                    else {

                                        System.out.println("\nDetails of deleted manufacturer:- \n");
                                        man.manufacturerDetails();
                                        //after manufacturer is deleted, products created by it will have no manufacturer
                                        Iterator<Product> it = man.set.iterator();
                                        while (it.hasNext()){

                                            Product p = it.next();
                                            p.man = null;
                                            this.hprod.replace(p.name,p);   //update changes to product in product's list
                                        }
                                        System.out.println("\nRemaining manufacturers are: "+this.hman.keySet());
                                    }
                            }

                            break;

                case 'b' :  if (this.hcust.size()==0)
                                    System.out.println("\nNo customers left.");
                                
                            else {
                                
                                    System.out.println("\nThe available customers are: "+this.hcust.keySet());
                                    System.out.print("Now enter name of customer to be deleted: ");
                                    name = in.nextLine();

                                    //remove selected customer from customer's list
                                    Customer cust = this.hcust.remove(name);
                                    if (cust == null)
                                        System.out.println("This customer does not exist");

                                    else {    

                                        System.out.println("\nDetails of deleted customer:- \n");
                                        cust.customerDetails();
                                        System.out.println("\nRemaining customers are: "+this.hcust.keySet());
                                    }
                            }

                            break;

                case 'c' :  if (this.hprod.size()==0)
                                    System.out.println("\nNo products left.");  

                            else {

                                    System.out.println("\nThe available products are: "+this.hprod.keySet());
                                    System.out.print("Now enter name of product to be deleted: ");
                                    name = in.nextLine();

                                    //remove selected product from product's list
                                    Product p = this.hprod.remove(name);
                                    if (p == null)
                                        System.out.println("This product does not exist");

                                    else {

                                        System.out.println("\nDetails of deleted product:- \n");
                                        p.productDetails();
                                        System.out.println("\nRemaining products are: "+this.hprod.keySet());

                                        //delete product from its manufacturer's list of created products
                                        Manufacturer man = this.hman.get(p.man.name);
                                        man.set.remove(p);
                                        this.hman.replace(man.name,man);

                                        //removing product from the inventory of all shops which had it
                                        for (Map.Entry<String,Shop> entry: this.hshop.entrySet()){

                                            Shop shop = entry.getValue();
                                            for(int i=0;i<shop.list.size();i++){

                                                Tuple tup = shop.list.get(i);
                                                if (p.name.equals(tup.p.name)){

                                                    shop.list.remove(tup);
                                                    break;
                                                }
                                            }
                                        }
                                    }
                            }

                            break;

                case 'd' :  if (this.hshop.size()==0)
                                    System.out.println("\nNo shops left."); 
                
                            else {

                                    System.out.println("\nThe available shops are: "+this.hshop.keySet());
                                    System.out.print("Now enter name of shop to be deleted: ");
                                    name = in.nextLine();

                                    //remove selected shop from shop's list
                                    Shop shop = this.hshop.remove(name);
                                    if (shop == null)
                                        System.out.println("This shop does not exist");

                                    else {

                                        System.out.println("\nDetails of deleted shop:- \n");
                                        shop.shopDetails();
                                        System.out.println("\nRemaining shops are: "+this.hshop.keySet());
                                    }
                            }

                            break;

                case 'e' :  if (this.hware.size()==0)
                                    System.out.println("\nNo warehouses left.");
                
                            else {

                                    System.out.println("\nThe available warehouses are: "+this.hware.keySet());
                                    System.out.print("Now enter name of warehouse to be deleted: ");
                                    name = in.nextLine();

                                    //remove selected warehouse from warehouse's list
                                    Warehouse ware = this.hware.remove(name);
                                    if (ware == null)
                                        System.out.println("This warehouse does not exist");

                                    else {

                                        System.out.println("\nDetails of deleted warehouse:- \n");
                                        ware.warehouseDetails();
                                        System.out.println("\nRemaining warehouses are: "+this.hware.keySet());
                                    }
                            }

                            break;   
                            
                case 'f'  : if (this.hdelv.size()==0)
                                    System.out.println("\nNo delivery agents left.");

                            else {

                                    System.out.println("\nThe available delivery agents are: "+this.hdelv.keySet());
                                    System.out.print("Now enter name of delivery agent to be deleted: ");
                                    name = in.nextLine();

                                    //remove selected delivery agent from delivery agent's list
                                    DeliveryAgent delv = this.hdelv.remove(name);
                                    if (delv == null)
                                        System.out.println("This delivery agent does not exist");

                                    else {

                                        System.out.println("\nDetails of deleted delivery agent:- \n");
                                        delv.deliveryAgentDetails();
                                        System.out.println("\nRemaining delivery agents are: "+this.hdelv.keySet());
                                    }
                            }

                            break;

                default   : System.out.println("You entered an invalid option, please try again!");
            }

            System.out.print("\n\nPlease enter your choice: ");
            choice = in.next().charAt(0);
            in.nextLine();
        }

        System.out.println("\n\nProcess of deleting objects complete.");
    }

//function to add product to a manufacturer
    public void addProdManufacturer(Scanner in){

        if (this.hman.size()==0)
            System.out.println("\nNo manufacturers left.");

        else {  

                System.out.println("\nThe available manufacturers are: "+this.hman.keySet());
                System.out.print("Now enter name of a manufacturer: ");
                String manName = in.nextLine();

                Manufacturer man = this.hman.get(manName);
                if (man == null)
                    System.out.println("This manufacturer does not exist");

                else {    
                
                    man.manufacturerDetails();
                    System.out.println("\nThe available products are: "+this.hprod.keySet());
                    System.out.print("Now enter name of a product: ");
                    String prodName = in.nextLine();
                    Product p = this.hprod.get(prodName);
                    if (p == null){

                        System.out.println("This product does not exist");
                        return;
                    }

                    man.addProduct(p);                  //assign product to manufacturer's list of created products
                    p.addManufacturer(man);                        //assign manufacturer to product
                    this.hman.replace(manName,man);
                    this.hprod.replace(p.name,p);       //update both in manufacturer's and product's list

                    if (this.hman.get(manName).set.contains(p)==false)
                        System.out.println("\nProduct "+p.name+" could not be added to manufacturer "+man.name);
                 
                    else {    

                        System.out.println("\nProduct "+p.name+" successfully added to manufacturer "+man.name+".");
                        System.out.println("\nPRODUCT DETAILS:-");
                        p.entityDetails();
                        System.out.println("\nPRODUCT MANUFACTURER DETAILS:-");
                        p.man.manufacturerDetails();
                    }
                }

        }    
    }

//function to add copies of product to a shop
    public void addProdShop(Scanner in){

        if (this.hshop.size()==0)
            System.out.println("\nNo shops left");

        else{
            
            System.out.println("\nThe available shops are: "+this.hshop.keySet());
            System.out.print("Now enter name of a shop: ");
            String shopName = in.nextLine();

            Shop shop = this.hshop.get(shopName);
            if (shop == null)
                System.out.println("This shop does not exist");

            else {

                System.out.println("\nSHOP DETAILS:-");
                shop.shopDetails();
                System.out.println("\nThe available products are: "+this.hprod.keySet());
                System.out.print("Now enter name of a product: ");
                String prodName = in.nextLine();
                Product p = this.hprod.get(prodName);
                if (p == null){

                    System.out.println("This product does not exist");
                    return;
                }

                System.out.print("Enter the number of products to be added: ");
                int n = in.nextInt();
                in.nextLine();
                for (int i=0;i<shop.list.size();i++){   //iterate over shop's inventory list

                    Tuple tup = shop.list.get(i);
                    if (tup.p.name.equals(prodName)){   //if product to be added already present,
                                                        //update its number in inventory
                        shop.list.remove(tup);
                        tup.number += n;
                        shop.list.add(tup);
                        this.hshop.replace(shopName, shop);         
                        System.out.println("\n"+n+" copies of product "+p.name+" were added successfully to shop "+shop.name+".");
                        shop.shopDetails();
                        return;
                    }

                }
                Tuple tup = new Tuple();                //if not present, create a new tuple and add to shop's inventory
                tup.p = p;
                tup.number = n;
                shop.list.add(tup);
                this.hshop.replace(shopName, shop);

                //checking if product was successfully added to shop inventory
                if (this.hshop.get(shopName).list.contains(tup)==false)
                    System.out.println("\n"+n+" copies of product "+p.name+" could not be added to shop "+shop.name);

                else{

                    System.out.println("\n"+n+" copies of product "+p.name+" were added successfully to shop "+shop.name+".");
                    shop.shopDetails();
                }
            }
  
        }
    }

//function to add an order of a product from a customer
    public void addProdCustomer(Scanner in){

        if (this.hcust.size()==0)           //if no customers left
            System.out.println("\nNo customers left");

        else{

            System.out.println("\nThe available customers are: "+this.hcust.keySet());
            System.out.print("Now enter name of a customer: ");
            String custName = in.nextLine();

            if (this.hcust.containsKey(custName) == false)      //if given customer does not exist
                System.out.println("This customer does not exist");

            else {    

                Customer cust = this.hcust.get(custName);
                System.out.println("\nCUSTOMER DETAILS:-");
                cust.customerDetails();
                System.out.println("\nThe available products are: "+this.hprod.keySet());
                System.out.print("Now enter name of a product: ");
                String prodName = in.nextLine();
                
                if (this.hprod.containsKey(prodName) == false){

                    System.out.println("This product does not exist");      //if given product does not exist
                    return;
                }
            
                Product p = this.hprod.get(prodName);
                p.productDetails();                                         //print details of given product
                System.out.print("Enter qty. of products needed: ");
                int qty = in.nextInt();
                in.nextLine();
                Order order = new Order(custName,prodName,qty);
                this.orderList.add(order);                                  //add given order to list of unprocessed orders

                if (this.orderList.contains(order)==false)                  //checking if order successfully added to list or not
                    System.out.println("Order of product "+p.name+" could not be added from customer "+cust.name);

                else
                    System.out.println("Order of product "+p.name+" successfully added from customer "+cust.name);
                  
            } 
        }
    }

//this function returns the delivery agent with the least number of delivered orders
    public DeliveryAgent findDeliveryAgent(){

        if (this.hdelv.size()==0){                          //if no delivery agents left
            System.out.println("No delivery agents left");
            return null;
        }

        Iterator<Map.Entry<String,DeliveryAgent>> it = this.hdelv.entrySet().iterator();
        int min = Integer.MAX_VALUE;
        DeliveryAgent minDel = new DeliveryAgent();

        //find the delivery agent with the least number of orders delivered
        while (it.hasNext()){

            Map.Entry<String,DeliveryAgent> entry = it.next();
            DeliveryAgent delv = entry.getValue();
            if (delv.delivered < min){

                min = delv.delivered;
                minDel = delv;
            }
        }

        return minDel;
    }

//function to process orders added previously and complete the order if feasible
//else cancel order due to availability / location issues
    public void processOrder(Scanner in){

        System.out.print("Enter name of customer: ");
        String custName = in.nextLine();
        System.out.print("Enter name of product: ");
        String prodName = in.nextLine();
        int i,qty=0;
        int lsize=this.orderList.size();    //stores initial size of list of unprocessed orders
                                            //needed bcoz if asked order found, it will be removed from list of orders, reducing list size
        for (i=0;i<this.orderList.size();i++){

            Order order = this.orderList.get(i);  
            if (custName.equals(order.custName) && prodName.equals(order.prodName)){   //if asked order found in list

                qty = order.qty;            //stores quantity of product to be bought
                this.orderList.remove(i);   //processed order is removed from list
                break;
            }
                
        }

        if (i==lsize)
            System.out.println("Order not found!");             

        else {

            Customer cust = this.hcust.get(custName);  //find customer from customers list
            int zipcode = cust.zip;
            Product p = this.hprod.get(prodName);      //find product from products list
            int count=0;

            Iterator<Map.Entry<String,Shop>> it = hshop.entrySet().iterator();

            while (it.hasNext()){                      //iterate over shops in shops list to find suitable shop

                Map.Entry<String,Shop> entry = it.next();
                Shop shop = entry.getValue();
                
                if (shop.zip != zipcode){              //if current shop's zipcode doesn't match with customer's zip,
                                                       //order cannot be processed
                    count++;
                    continue;
                }

                for (int j=0;j<shop.list.size();j++){  //iterate over products in shop inventory

                    Tuple tup = shop.list.get(j);
                    if (tup.p.name.equals(p.name) && tup.p.id==p.id){   //if asked product is available in shop

                        if (tup.number-qty <0)              //if asked amount is more than that available with shop,
                            continue;                       //order cannot be processed

                        System.out.println("Order of "+p.name+" for customer "+cust.name+" has been processed successfully.");
                        
                        cust.list.add(p);                           //update customer's list of purchases
                        this.hcust.replace(cust.name,cust);         //update changes in customers list
                        
                        //update changes in shop inventory
                        shop.list.remove(tup);              
                        tup.number = tup.number - qty;         
                        shop.list.add(tup);
                        this.hshop.replace(shop.name,shop);         //update changes in shops list
                        System.out.println("DETAILS OF SHOP:-\n");  //print shop details
                        shop.shopDetails();

                        //print delivery agent details
                        System.out.println("DETAILS OF DELIVERY AGENT:-\n");
                        DeliveryAgent delv = findDeliveryAgent();
                        delv.delivered++;
                        delv.deliveryAgentDetails();                //update no. of deliveries by agent and
                        hdelv.replace(delv.name,delv);              //update changes in delivery agents list
                        return;
                    }
                }
                count++;
            }
            

            if (count == hshop.size()) //print message if order couldn't be processeddue to availability / location issues
                System.out.println("Order could not be processed due to availability / location issues.");
        }    
    }


//function to print purchase details of customer
    public void printPurchaseDetails(Scanner in){

        if (this.hcust.size()==0)
            System.out.println("\nNo customers left.");

        else{
            
            System.out.println("\nThe available customers are: "+this.hcust.keySet());
            System.out.print("Now enter name of a customer: ");
            String name = in.nextLine();

            Customer cust = this.hcust.get(name);
            if (cust == null)
                System.out.println("This customer does not exist");

            else{

                //print purchase details of customer
                System.out.println("\nPURCHASE DETAILS OF CUSTOMER "+cust.name+" :-\n");
                for (int i = 0 ; i < cust.list.size() ; i++ ){

                    System.out.println((i+1)+".     "+cust.list.get(i).name);
                }
            }

        }
    }

//print details of shop inventory
    public void shopInventory(Scanner in){

        if (this.hshop.size()==0)
            System.out.println("\nNo shops left.");

        else{

            System.out.println("\nThe available shops are: "+this.hshop.keySet());
            System.out.print("Now enter name of a shop: ");
            String name = in.nextLine();

            Shop shop = this.hshop.get(name);
            if (shop == null)
                System.out.println("This shop does not exist");

            else     
                shop.shopDetails();         //print shop details
        }
    }

//function to print all products made by a manufacturer
    public void manufactProduct(Scanner in){

        if (this.hman.size()==0)
            System.out.println("\nNo manufacturers left.");

        else{

            System.out.println("\nThe available manufacturers are: "+this.hman.keySet());
            System.out.print("Now enter name of a manufacturer: ");
            String name = in.nextLine();

            Manufacturer man = this.hman.get(name);
            if (man == null)
                System.out.println("This manufacturer does not exist");
            else    
                man.manufacturerDetails();          //print products made by manufacturer
        }
    }

//The main method 
    public static void main(String args[]){

        MedicineStore medStore = new MedicineStore();   //create a new MedicineStore object to access funtionalities offered by it
        Scanner in = new Scanner(System.in);
        System.out.println("--------------------------------------\n");
        System.out.println("WELCOME TO THE ONLINE MEDICINE STORE!");
        System.out.println("Please have a look at the following options ...\n");
        medStore.displayMenu();                         //print the display menu to show available funtionalities
        System.out.print("Please enter your choice: "); 
        int choice = in.nextInt();
        in.nextLine();

        while (choice!=0) {

            switch(choice){

                case 1 :    //creating and deleting entities
                            System.out.println("\n**********CREATING AND DELETING ENTITIES**********");
                            //provide 3 user-defined options to create, delete and print an entity of any type
                            System.out.println("\nPress C to create entities");
                            System.out.println("Press D to delete entities");
                            System.out.println("Press P to print entities");
                            System.out.println("Press Q to quit");
                            System.out.print("\nPlease enter your choice: ");
                            char option = in.next().charAt(0);
                            in.nextLine();

                            while (option!='Q') {

                                switch(option){

                                    case 'C' :  medStore.createEntities(in);    break;  //create entity of any type
                                    case 'D' :  medStore.deleteEntities(in);    break;  //delete entity of any type
                                    case 'P' :  medStore.printEntities(in);     break;  //print entity of any type
                                    default  :  System.out.println("You entered an invalid option, please try again!");
                                }

                                System.out.println("\n\nPress C to create entities");
                                System.out.println("Press D to delete entities");
                                System.out.println("Press Q to quit");
                                System.out.print("\nPlease enter your choice: ");
                                option = in.next().charAt(0);
                                in.nextLine();
                            }

                            break;

                case 2 :    //add a product to manufacturer
                            System.out.println("\n\n**********ADD A PRODUCT TO MANUFACTURER**********");
                            medStore.addProdManufacturer(in);
                            break;

                case 3 :    //add a given number of products to a shop
                            System.out.println("\n\n********ADD SOME NUMBER OF PRODUCTS TO SHOP********\n");
                            medStore.addProdShop(in);
                            break;

                case 4 :    //add order of a product from a customer 
                            System.out.println("\n\n********ADD ORDER OF PRODUCT TO CUSTOMER********\n");
                            medStore.addProdCustomer(in);
                            break;

                case 5 :    //process the already added orders for a customer
                            System.out.println("\n\n********PROCESS ORDERS OF CUSTOMER********\n");
                            medStore.processOrder(in);
                            break;
         
                case 6 :    //display the purchases made by a customer
                            System.out.println("\n\n********ENTER PURCHASES MADE BY CUSTOMER********\n");
                            medStore.printPurchaseDetails(in);
                            break;

                case 7 :    //list inventory details of a shop
                            System.out.println("\n\n********LIST INVENTORY OF SHOP********\n");
                            medStore.shopInventory(in);
                            break;

                case 8 :    //list the products made by a manufacturer
                            System.out.println("\n\n********PRODUCTS MADE BY MANUFACTURER********\n");
                            medStore.manufactProduct(in);
                            break;

                default :   System.out.println("You entered an invalid option, please try again!");
                  
            }

            medStore.displayMenu();
            System.out.print("\n\nPlease enter your choice: ");
            choice = in.nextInt();
            in.nextLine();
        }

        System.out.println("\nTHANK YOU FOR SHOPPING WITH US! PLEASE VISIT AGAIN.");
        System.out.println("\n--------------------------------------");
        in.close();
    }
}

//class created to store a tuple of (Product,Integer) type
class Tuple{

    Product p;
    int number;
}

//class created to store an order with customer name, product name and qty needed
class Order{

    String custName ;
    String prodName ;
    int qty;

    public Order(String custName, String prodName,int qty){

        this.custName = custName;
        this.prodName = prodName;
        this.qty = qty;
    }
}

//the basic class which defines common properties for all entities
class Entity{

    int id;
    String name;
    public Entity(){

        this.id=0;
        this.name="";
    }

    public Entity(Scanner in){

        System.out.print("Enter name: ");
        this.name = in.nextLine();
        System.out.print("Enter unique id: ");
        this.id = in.nextInt();
        in.nextLine();
    }

    public void entityDetails(){            //general method to print details of any entity

        System.out.println("\nName: "+this.name);
        System.out.println("Unique ID: "+this.id);
    }
}

//class to define a product
class Product extends Entity{

    Manufacturer man;

    public Product(Scanner in){

        super(in);
    }

    public void addManufacturer(Manufacturer man){        //method to assign manufacturer to product

        this.man=man;
    }

    public void productDetails(){          //method to print details of a customer

        this.entityDetails();
        if (man == null)    
            return;
        System.out.println("\nDETAILS OF PRODUCT MANUFACTURER:-\n");
        this.man.manufacturerDetails();
    }
}

//class to define a manufacturer
class Manufacturer extends Entity{

    HashSet<Product> set;

    public Manufacturer(Scanner in){

        super(in);
        this.set=new HashSet<Product>();
    }

    public void addProduct(Product p){      //function to add a product to a manufacturer

        this.set.add(p);
    }

    public void manufacturerDetails(){      //function to print details of a manufacturer

        this.entityDetails();
        System.out.println("\nDETAILS OF PRODUCTS BY "+this.name+" :-\n");
        Iterator<Product> it = this.set.iterator();
        int i=0;

        while (it.hasNext())
            System.out.println(++i+". "+it.next().name);

    }
}

//class to define a customer
class Customer extends Entity{

    int zip;
    List<Product> list;

    public Customer(Scanner in){

        super(in);
        System.out.print("Enter zip code: ");
        this.zip = in.nextInt();
        in.nextLine();
        this.list=new ArrayList<>();
    }

    public void customerDetails(){         //method to print details of a customer

        this.entityDetails();
        System.out.println("Zip code: "+this.zip);
        System.out.println("List of products ordered: \n");
        for (int i=0;i<this.list.size();i++){

            System.out.println((i+1)+". "+this.list.get(i).name);
        }

    }
}


class Shop extends Entity{

    int zip;
    List<Tuple> list;

    public Shop(Scanner in){

        super(in);
        System.out.print("Enter zip code: ");
        this.zip = in.nextInt();
        in.nextLine();
        this.list = new ArrayList<>();
    }

    public void shopDetails(){      //method to print details of a shop

        this.entityDetails();
        System.out.println("Zip code: "+this.zip);
        System.out.println("\nDETAILS OF INVENTORY OF "+this.name+" :-\n");
        for (int i=0; i<this.list.size(); i++){

            System.out.println((i+1)+". "+this.list.get(i).p.name+"             "+this.list.get(i).number);
        }
    }
}


class Warehouse extends Entity{

    int zip;
    List<Tuple> list;

    public Warehouse(Scanner in){

        super(in);
        System.out.print("Enter zip code: ");
        this.zip = in.nextInt();
        in.nextLine();
        this.list = new ArrayList<Tuple>();
    }

    public void warehouseDetails(){         //method to print details of a warehouse

        this.entityDetails();
        System.out.println("Zip code: "+this.zip);
        System.out.println("\nDETAILS OF INVENTORY OF "+this.name+" :-\n");
        for (int i=0; i<this.list.size(); i++){

            System.out.println((i+1)+". "+this.list.get(i).p.name+"             "+this.list.get(i).number);
        }
    }
}


class DeliveryAgent extends Entity{

    int zip;
    int delivered;

    //here, constructor of DeliveryAgent class is overloaded to allow for two different types of initializations
    public DeliveryAgent(){

        super();
        this.zip=this.delivered=0;

    }
    public DeliveryAgent(Scanner in){

        super(in);
        System.out.print("Enter zip code: ");
        this.zip = in.nextInt();
        in.nextLine();
        this.delivered = 0;
    }

    public void deliveryAgentDetails(){         //method to print details of delivery agent

        this.entityDetails();
        System.out.println("Zip code: "+this.zip);
        System.out.println("No. of products delivered: "+this.delivered);
    }
}

