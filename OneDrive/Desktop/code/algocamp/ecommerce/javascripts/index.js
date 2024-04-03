// console.log("loaded");

async function fetchcategories(){
    //this function is marked async so this will also return a promise
const response= await fetch("https://fakestoreapi.com/products/categories");
const data= await response.json();
return data;
}
fetchcategories();

async function populatecategories(){
    const categories=await fetchcategories();
    removeloader();
    const categorylist =document.getElementById("categorylist");
    categories.forEach(category => {
        const categoryholder=document.createElement("div");
        const categorylink=document.createElement("a");
        categorylink.href=`productlist.html?category=${category}`
        categorylink.textContent= category //setting the category name as the text of the anchor tag
        
        categoryholder.classList.add("category-item" ,"d-flex", "flex-row", "justify-content-center", "align-items-center")
        categoryholder.appendChild(categorylink);
        categorylist.appendChild(categoryholder);
    
    });

}
populatecategories();

