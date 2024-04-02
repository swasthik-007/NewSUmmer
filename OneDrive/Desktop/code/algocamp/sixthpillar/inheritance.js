function product(n,p,c){
    this.name=n;
    this.price=p;
    category.call(this,c);
}
function category(c){
    this.categoryname=c;
    this.getcategoryname=function (){
        console.log((this.categoryname));
    }

}

let p= new product("iphone",10000,"electronics");
p.getcategoryname();