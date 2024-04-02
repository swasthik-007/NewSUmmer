function createPromise(){
    return new Promise(function exec(resolve,reject){
        setTimeout(function (){
            console.log("REJECTING the promise");
            reject("done");
        },1000);
    });
}

setTimeout(function process(){
    console.log("timer completed ");
},0 );

let p =createPromise();
p.then(function fulfillhandler1(Value){
    console.log("we fulfilled1 with a value",Value);
},function rejectionhandler(Value){
    console.log("we reject1 with a value",Value);
});
p.then(function fulfillhandler2(Value){
    console.log("we fulfilled2 with a value",Value);
},function rejectionhandler(Value){
    console.log("we reject2 with a value",Value);
});


for(let i=0;i<100000000;i++){}
console.log("ending");