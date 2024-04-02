const player={
    firstname: 'virat',
    lastname: 'kohli',
    number: 3,
    canbowl: false,
    getdetails:function (){
        console.log(this.firstname,this.lastname,"comes at",this.number);
    }
}

// const obj =function(){
//     console.log(this.getdetails());
// }

// let x=obj.bind(player);

// x();
const obj =function(x,y){
    console.log(x+y);
    this.getdetails();
}

obj.call(player,2,34);