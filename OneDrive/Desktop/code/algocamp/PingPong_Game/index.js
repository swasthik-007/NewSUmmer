document.addEventListener("DOMContentLoaded", () => {
    let ball=document.getElementById("ball") //targetting the ball element
    let table=document.getElementById("pingpongtable");
    let paddle=document.getElementById("paddle") //targetting the paddle element


    //here the  ballx and bally will be helping us to set a starting point of ball wrt table

    let ballx=50; //distance  of the top of the ball wrt pingpong table
    let bally=50; //distance of the left of the ball wrt ping pong table;
    
    let dx=2; //sdisplacement
    let dy=2;

    ball.style.top= `${bally}px`;
    ball.style.left= `${ballx}px`;
    setInterval(function exec(){

        ballx+=dx;
        bally+=dy;

        ball.style.left=`${ballx}px`;
        ball.style.top= `${bally}px`;

        let score=0;
        // if(ballx> 700-20 or ballx<=0) dx*=-1;
        // if(bally> 400-20 or bally<=0) dy*=-1;
        if(ballx< paddle.offsetLeft + paddle.offsetWidth && bally> paddle.offsetTop && bally+ ball.offsetHeight<paddle.offsetTop+ paddle.offsetHeight){
            dx*=-1;
            `${score}px`;

        }
        if(ballx>table.offsetWidth - ball.offsetWidth || ballx<=0) dx*=-1;
        if(bally>table.offsetHeight - ball.offsetHeight || bally<=0) dy*=-1;
    

        //collision of ball and paddle

    },1);

    let paddley=0;
    let dpy=10;

    document.addEventListener("keydown",(event)=>{
        event.preventDefault();  // prevents the execution oof the default behaviour

        if(event.keyCode==38 && paddley>0){
            //up arrow
            paddley+= (-1)*dpy;
        }else if(event.keyCode==40 &&  paddley<table.offsetHeight-paddle.offsetHeight){
            //down arrow
            paddley+=dpy;
        }
        paddle.style.top=`${paddley}px`;
            
    });

    document.addEventListener("mousemove",(event)=>{
        let mousedistancefromtop=event.clientY; //this is the distance of the mouse from the top
        let distanceoftanblefromtop=table.offsetTop;
        let mousepointcontrol=mousedistancefromtop-distanceoftanblefromtop-paddle.offsetHeight/2;
        paddley=mousepointcontrol;
        if(paddley<=0 || paddley>table.offsetHeight-paddle.offsetHeight) return;//if bottom of the paddle has passed 
    paddle.style.top=`${paddley}px`;

    })

});