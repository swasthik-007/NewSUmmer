class product{
    #name; //# lagane se private declare karte h
    #price;
    // desc;
     
    constructor(n,p,d){//consturctor are meant to return an object
        this.#name=n;
        this.price=p;
        this.desc=d;
       // return 10;//primitive-> no effect
        // return {};
        //if you dont return anything , it is equal to saying returm this
        // return this;
    }
    set name(n){
        if(typeof(n)!='string') {
         console.log("invalid name passed");
         return;
        }
        this.#name=n;
    }
    get name(){
        //print the product
        console.log(this.#name,this.#price,this.desc);
    }
    setprice(p)
            {
    if(p<0) {
        console.log("invalid");
        return;}
    this.#price=p;
            }    
    display(){
        //print the product
        console.log(this.#name, this.price,this.desc);
    }
            }

const p= new product("bag",100,"cool bag"); // it calls the default constructor alltogeht
p.name="swas";
// p.setname(-1);
p.price=1;
console.log(p);
// p.display();
