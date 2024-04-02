function createPromise(){
    return new Promise(function exec(resolve,reject){
        console.log("resoloving the promise");
        resolve("done");
    });
}

setTimeout(function process(){
    console.log("timer completed ");
},0 );

let p =createPromise();
p.then(function fulfillhandler1(Value){
    console.log("we fulfilled1 with a value",Value);
},function rejectionhandler(){});
p.then(function fulfillhandler2(Value){
    console.log("we fulfilled2 with a value",Value);
},function rejectionhandler(){});
p.then(function fulfillhandler3(Value){
    console.log("we fulfilled3 with a value",Value);
},function rejectionhandler(){});

for(let i=0;i<10000000000;i++){}
console.log("ending");
// at any point  of time if eveent loop has a choice to pick 
// from microtask queueor callback queue(macrotask queue) then it always gives preference to microtask queue