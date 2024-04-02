class product{
    name;
    price;
    desc;
     
    constructor(n,p,d){//consturctor are meant to return an object
        this.name=n;
        this.price=p;
        this.desc=d;
       // return 10;//primitive-> no effect
        // return {};
        //if you dont return anything , it is equal to saying returm this
        return this;
    }
    display(){
        //print the product
    }
}

const p= new product("bag",100,"cool bag"); // it calls the default constructor alltogeht
console.log(p);
