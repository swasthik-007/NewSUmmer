function product(n,p,d){
    this.name=n;
    this.price=p;
    this.description =d;
this.display=function()
{
    console.log(this);
}
}

const p=new product("bag" , 100 , "cool new bag");

p.display();