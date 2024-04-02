function download(url,time){
    return new Promise(function exec(resolve,reject){
        console.log("starting to download from ",url);
    setTimeout(function down(){
        console.log("downloading completed");
        const content="ABCDEF"; // assume dummy download content
       if(time> 3000){
        reject("err");
       }
       else
        resolve(content);
    },time);
    })
 }

 const p1=download("www.a1.com",15000);
 const p2=download("www.a2.com",6000);
 const p3=download("www.a3.com",1000);

 Promise.all([p1,p2,p3]).then(function fulfillhandler(value){
    console.log(value);
 })