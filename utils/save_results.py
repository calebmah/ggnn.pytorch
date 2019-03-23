# -*- coding: utf-8 -*-
from datetime import datetime
import xlsxwriter

def save_results(opt, results):
    TASK_IDS = [1, 2, 4, 9, 11, 12, 13, 15, 16, 17, 18]
    # Create a workbook and add a worksheet.
    name = datetime.now().strftime("%d-%m-%Y %H.%M.%S")
    workbook = xlsxwriter.Workbook('{} {} {} {} {} {} {} {}.xlsx'.format(name,opt.net, opt.train_size, opt.niter, opt.n_steps, opt.state_dim, opt.lr, opt.L))
    worksheet = workbook.add_worksheet()
    
    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': True})
    
    # Iterate over the data and write it out row by row.
    worksheet.write(0, 0, name)
    worksheet.write(1, 0, "Parameters", bold)
    worksheet.write(1, 3, "Task", bold)
    worksheet.write(1, 4, "Average Loss", bold)
    worksheet.write(1, 6, "Accuracy", bold)
    worksheet.write(1, 7, "Time", bold)
    for row, item in enumerate(["net","cuda","train_size", "niter", "n_steps","state_dim","lr","D","H","L"],2):
        worksheet.write(row, 0, item)
        worksheet.write(row, 1, vars(opt)[item])
    
    row = 2
    col = 3 
    
    for i, (train_loss, test_loss, numerator, denominator, clock) in enumerate(results):
        worksheet.write(row, col,     TASK_IDS[i])
        worksheet.write(row, col + 1, train_loss)
        worksheet.write(row, col + 2, "{}/{}".format(numerator,denominator))
        worksheet.write(row, col + 3, numerator.item()/denominator, workbook.add_format({'num_format': '0.00%'}))
        worksheet.write(row, col + 4, clock)
        row += 1
    
    workbook.close()
    print('{}.xlsx saved'.format(name))