const salaryData = require('./salaryData.js').salaryData;
const getDataByRole = require('./salaryData.js').getDataByRole;
const getDataByCompany = require('./salaryData.js').getDataByCompany;

// Replace the empty array with the appropriate imported function/value
const getAverageSalaryByRole = role => {
  const roleData = getDataByRole(salaryData);
  const salariesOfRole = roleData.map(obj => obj.salary);
  return calculateAverage(salariesOfRole);
};

// Replace the empty array with the appropriate imported function/value
const getAverageSalaryByCompany = company => {
  const companyData = getDataByCompany(salaryData);
  const salariesAtCompany = companyData.map(obj => obj.salary);
  return calculateAverage(salariesAtCompany);
};

// Replace the empty array with the appropriate imported function/value
const getSalaryAtCompany = (role, company) => {
  const companyData = getDataByCompany(salaryData);
  const roleAtCompany = companyData.find(obj => obj.role === role);
  return roleAtCompany.salary;
};

// Replace the empty array with the appropriate imported function/value
const getIndustryAverageSalary = () => {
  const allSalaries = getDataByRole(salaryData).map(obj => obj.salary);
  return calculateAverage(allSalaries);
};


// Helper Function. Do not edit.
// Note: This function does not need to be exported since it is only used by the functions contained within this module.
function calculateAverage(arrayOfNumbers) {
  let total = 0;
  arrayOfNumbers.forEach(number => total += number);
  return (total / arrayOfNumbers.length).toFixed(2);
};

module.exports = {
    getAverageSalaryByRole,
    getAverageSalaryByCompany,
    getSalaryAtCompany,
    getIndustryAverageSalary
};
