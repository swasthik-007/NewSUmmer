// let gettodosbutton=document.getElementById('get-todos');
// let tododetails=document.getElementsByClassName('todo-detail');
// // gettodosbutton.addEventListener("click",() => {
// //     console.log("clicked");
// // })
// // tododetails.addEventListener("click",() => {// wont happen bcz to do detail is an arrray and not a function
// //     console.log("clicked");
// // })

// // another way of clicking is 
// //  gettodosbutton.onclick= () =>{
// //     console.log("clicked");
// //  }

//  //another way is
//  //write an attribute in html using onclick and write one function onclick
//  function clickbtn(){
//     console.log("clicked");
//  }

// console.log("welcome to my to do app");
let  todos=[];
let tododatalist=document.getElementById("todo-data-list");
let savebutton=document.getElementById("save-todo");
let todoinputbar=document.getElementById("todo-input-bar");
let getpendingtodobutton=document.getElementById("get-todos");

getpendingtodobutton.addEventListener("click",()=>{
    todos=todos.filter((todo)=>todo.status !="Finished");
    rerendertodo();
})
todoinputbar.addEventListener("keyup",function togglesavebutton(){
     let todotext=todoinputbar.value;
     if(todotext.length==0){
        if(savebutton.classList.contains("disabled")) return;
        savebutton.classList.add("disabled");
     }
     else if(savebutton.classList.contains("disabled")){
     savebutton.classList.remove("disabled");
     }
})
savebutton.addEventListener("click", function gettextandaddtodo(){
    let todotext=todoinputbar.value;
    if(todotext.length==0) return;
    let todo={text: todotext,status : 'In Progress', finishbuttontext: 'finished'};
    todos.push(todo);
    addtodo(todo,todos.length);
    todoinputbar.value='';
})
function rerendertodo(){
    tododatalist.innerHTML=''; 
    todos.forEach((element,idx)=>{
      addtodo(element,idx+1);
    });
}
function removetodo(event){
    // console.log("clicked",event.target.parentElement.parentElement.parentElement);
    // event.target.parentElement.parentElement.parentElement.remove();
    let deletebuttonpressed=event.target;
    let indextoberemoved=Number(deletebuttonpressed.getAttribute("todo-idx"));
    todos.splice(indextoberemoved,1)
    rerendertodo();
}
function finishtodo(event){
    let finishbuttonpressed=event.target;
    let indextobefinished=Number(finishbuttonpressed.getAttribute("todo-idx"));
   //toggling
    if(todos[indextobefinished].status=="Finished"){
    todos[indextobefinished].status="In Progress";
    todos[indextobefinished].finishbuttontext="Finished";
   }
   else{
    todos[indextobefinished].status="Finished";
    todos[indextobefinished].finishbuttontext="Undo";
   }
   todos.sort((a,b)=>{
    if(a.status=="Finished"){
        return 1;

    }
    return -1;
   })
    // todos[indextobefinished].status="Finished";
    // todos[indextobefinished].finishbuttontext="Undo";
    rerendertodo();
}

function edittodo(event){
    let editbuttonpressed=event.target;
    let indextoedit=Number(editbuttonpressed.getAttribute("todo-idx"));
    let detaildiv=document.querySelector(`div[todo-idx="${indextoedit}"]`)
    let input=document.querySelector(`input[todo-idx="${indextoedit}"]`)
    detaildiv.style.display="none";
    input.type="text";
    input.value=detaildiv.textContent;

}
function saveedittodo(event){
    let input=event.target;
    let indextoedit=Number(input.getAttribute("todo-idx"));
    let detaildiv=document.querySelector(`div[todo-idx="${indextoedit}"]`)
if(event.keyCode==13){

    detaildiv.textContent=input.value;
    detaildiv.style.display="block";
    input.value='';
    input.type="hidden";
}
}
function addtodo( todo , todocount ){
    let rowdiv=document.createElement("div");
    let todoitem=document.createElement("div");
    let todonumber=document.createElement("div");
    let tododetail=document.createElement("div");
    let todostatus=document.createElement("div");
    let todoactions=document.createElement("div");
    let deletebutton=document.createElement("button");
    let finishedbutton=document.createElement("button");
    let editbutton=document.createElement("button");
    let hiddeninput=document.createElement("input");
    let hr=document.createElement("hr");


     //adding classes
     rowdiv.classList.add("row");
     todoitem.classList.add("todo-item", "d-flex" ,"flex-row" ,"justify-content-between", "align-items-center");
     todonumber.classList.add("todo-no");
     tododetail.classList.add("todo-detail", "text-muted")
     todostatus.classList.add("todo-status","text-muted")
     todoactions.classList.add("todo-action","d-flex","justify-content-start","gap-2");
     deletebutton.classList.add("btn","btn-danger","delete-todo");
     finishedbutton.classList.add("btn","btn-success","finish-todo");
     editbutton.classList.add("btn","btn-warning","edit-todo");
    hiddeninput.classList.add("form-control","todo-detail");

    
     finishedbutton.setAttribute("todo-idx",todocount-1);
    deletebutton.setAttribute("todo-idx",todocount-1);
    editbutton.setAttribute("todo-idx",todocount-1);
    tododetail.setAttribute("todo-idx",todocount-1);
    hiddeninput.setAttribute("todo-idx",todocount-1);
    hiddeninput.addEventListener("keypress",saveedittodo)
   
    deletebutton.onclick=removetodo;
    finishedbutton.onclick=finishtodo;
    editbutton.onclick= edittodo;
    hiddeninput.type="hidden";
    



    todonumber.textContent=`${todocount}.`;
    tododetail.textContent=todo.text;//sets the todotext sent from the input
    todostatus.textContent=todo.status;
    deletebutton.textContent="Delete";
    finishedbutton.textContent=todo.finishbuttontext;
    editbutton.textContent="edit";

    todoactions.appendChild(deletebutton);
    todoactions.appendChild(finishedbutton);
    todoactions.appendChild(editbutton);
    

    todoitem.appendChild(todonumber);
    todoitem.appendChild(tododetail);
    todoitem.appendChild(hiddeninput);
    todoitem.appendChild(todostatus);
    todoitem.appendChild(todoactions);

    rowdiv.appendChild(todoitem);
    rowdiv.appendChild(hr);

    tododatalist.appendChild(rowdiv);

}