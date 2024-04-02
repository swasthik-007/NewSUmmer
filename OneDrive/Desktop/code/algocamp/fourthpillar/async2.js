function timeconsumingbyloop(){
    console.log("loop starts");
    for(let i=0;i<1000000000;i++){
        //sometask
    }
    console.log("loop ends");
}
function timeconsumingbyruntimefeature(){
    console.log("starting timer");
    setTimeout(function exec(){
        console.log("completed the timer");
    },5000);
}

console.log("hi");

timeconsumingbyloop();
timeconsumingbyruntimefeature();
timeconsumingbyloop();

console.log("bye");