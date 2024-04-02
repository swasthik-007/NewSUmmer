let iphone={
    name: "iphone",
    price: 10000,
    description:" new one",
    rating : 4.8,

    display(){
        console.log("first display",this);
    }
}
let macbook={
    name: "iphone",
    price: 10000,
    description:" new one",
    rating : 4.8,

    display(){
        console.log(this);
    }
}

// iphone.display();

let products={
    arr:[
    {
        name: "iphone",
        price: 10000,
        description:" new one",
        rating : 4.8,
        getcategory:()=>{console.log(this.category);}
    
    },
    {
        name: "macbook",
        price: 55000,
        description:" laptop one",
        rating : 4.7,
        getcategory: ()=> {console.log(this.category);}
    
    }
],
category:"Electronics",
getproducts:function(){
    return this.arr;
}
}

products.getproducts()[0].getcategory();