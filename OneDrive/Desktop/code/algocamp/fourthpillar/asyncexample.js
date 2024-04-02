function timeconsumingbyloop(){
    console.log("loop starts");
    for(let i=0;i<10000000000;i++){
        //some task
    }
    console.log("loop ends");
}
function timeconsumingbyruntimefeature0(){
    console.log("starting timer");
    setTimeout(function exec(){
        console.log("completed timer0");
        for(let i=0;i<10000000000;i++){
            //some task
        }
    },5000);//5sec timer
}

function timeconsumingbyruntimefeature1(){
    console.log("starting timer");
    setTimeout(function exec(){
        console.log("completed the timer 1");

    },0); // 0s timer
}

function timeconsumingbyruntimefeature2(){
    console.log("starting timer");
    setTimeout(function exec(){
        console.log("completed the timer 2");
    },200); //200ms timer
}

console.log("hi");
timeconsumingbyloop();
timeconsumingbyruntimefeature0();
timeconsumingbyruntimefeature1();
timeconsumingbyruntimefeature2();
timeconsumingbyloop();
console.log("bye");

//console.log is not core java feature
