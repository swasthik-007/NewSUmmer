var obj={
    name: "sanket",
    company:"google",
    display:()=>{
        console.log(this.name,"works in",this.company);
    }
}
var obj1 ={
    name: "JD",
    company:"Microsoft",
    display:()=>{
      setTimeout(()=>{

          console.log(this.name,"works in",this.company);
        },3000);
    }
}
var obj2 ={
    name: "sarthak",
    company:"Phonepe",
    display:function (){
        setTimeout(()=>{
  
            console.log(this.name,"works in",this.company);
          },3000);
      }
}

obj2.display();