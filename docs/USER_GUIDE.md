# PV Circularity Simulator - User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Creating Your First Project](#creating-your-first-project)
3. [Timeline Planning](#timeline-planning)
4. [Resource Management](#resource-management)
5. [Contract Management](#contract-management)
6. [Portfolio Management](#portfolio-management)
7. [Tips and Best Practices](#tips-and-best-practices)

## Getting Started

### Launching the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Navigation

The sidebar provides navigation to all main features:
- **🏠 Home**: Overview and quick start
- **📊 Portfolio Dashboard**: Multi-project view
- **🧙 Project Wizard**: Create new projects
- **📅 Timeline Planner**: Manage timelines
- **📦 Resource Allocation**: Manage resources
- **📄 Contract Management**: Manage contracts

## Creating Your First Project

### Using the Project Wizard

The Project Wizard guides you through a 4-step process:

#### Step 1: Basic Information
1. Enter **Project Name** (required)
2. Add a **Description** of your project
3. Specify the **Project Owner/Manager** (required)
4. Enter the **Installation Location** (required)

Click **Next →** to continue.

#### Step 2: Technical Specifications
1. Enter **System Capacity** in kWp (required)
2. Select **Initial Project Status** (default: Design)
3. Optionally add:
   - PV Module Type (e.g., "Monocrystalline")
   - Inverter Type (e.g., "String Inverter")
   - Estimated Module Count
   - Mounting System type

Click **Next →** to continue.

#### Step 3: Timeline & Budget
1. Select **Project Start Date** (required)
2. Select **Expected Completion Date** (required)
3. Enter **Total Project Budget** in USD (required)
4. Optionally add up to 2 key milestones with dates

Click **Next →** to continue.

#### Step 4: Review & Create
1. Review all entered information
2. Click **✓ Create Project** to finalize
3. Your project is now created and saved!

### After Creation

After creating a project:
- You'll see a success message with balloons 🎈
- The project is automatically saved to persistent storage
- You can now add timelines, resources, and contracts

## Timeline Planning

### Accessing Timeline Planner

1. Navigate to **📅 Timeline Planner**
2. Select your project from the dropdown
3. View three tabs: **Gantt Chart**, **Milestones**, **Phases**

### Working with the Gantt Chart

The Gantt Chart provides a visual overview of:
- Overall project timeline
- Project phases with durations
- Milestones as zero-duration markers

The chart updates automatically as you add milestones and phases.

### Adding Milestones

1. Go to the **Milestones** tab
2. Click **➕ Add New Milestone**
3. Enter:
   - Milestone Name (required)
   - Target Date (required)
   - Description (optional)
4. Click **Add Milestone**

### Managing Milestones

For each milestone you can:
- **Mark Complete**: Click the button to mark as completed
- **Delete**: Remove the milestone from the timeline
- View completion status (✓ for complete, ○ for pending)

### Adding Phases

1. Go to the **Phases** tab
2. Click **➕ Add New Phase**
3. Enter:
   - Phase Name (required)
   - Start Date (required)
   - End Date (required, must be after start date)
   - Description (optional)
4. Click **Add Phase**

### Example Phases

Common PV project phases:
- Design Phase (1-2 months)
- Permitting (2-3 months)
- Procurement (1-2 months)
- Installation (2-4 months)
- Commissioning (1 month)

## Resource Management

### Accessing Resource Dashboard

1. Navigate to **📦 Resource Allocation**
2. Choose to filter by project or view all resources
3. View summary metrics at the top

### Understanding Metrics

- **Total Resources**: Number of resource entries
- **Allocated**: Resources marked as allocated to projects
- **Total Cost**: Sum of all resource costs
- **Budget Used**: Percentage of project budget consumed (when filtered by project)

### Adding Resources

1. Go to the **Add Resource** tab
2. Select the project (if not already filtered)
3. Enter resource details:
   - **Resource Name** (required)
   - **Resource Type** (Module, Inverter, Cable, Labor, etc.)
   - **Quantity** (required)
   - **Unit** (pieces, hours, kW, etc.)
   - **Unit Cost** (required)
   - **Supplier/Vendor**
4. Optionally set:
   - Availability window (from/to dates)
   - Constraints (one per line)
   - Allocation status
5. Click **Add Resource**

### Resource Types

Available resource types:
- **PV Module**: Solar panels
- **Inverter**: DC to AC converters
- **Cable**: Electrical wiring
- **Mounting System**: Racking and support structures
- **Labor**: Installation and service hours
- **Capital**: Financial resources
- **Equipment**: Tools and machinery
- **Other**: Miscellaneous resources

### Viewing Analytics

The **Analytics** tab provides:
- **Cost Breakdown by Type**: Pie chart showing cost distribution
- **Allocation Status**: Bar chart of allocated vs unallocated
- **Top Suppliers**: Bar chart of top 5 suppliers by cost

## Contract Management

### Accessing Contract Management

1. Navigate to **📄 Contract Management**
2. Choose to filter by project or view all contracts
3. View summary metrics

### Contract Metrics

- **Total Contracts**: Number of contracts
- **Active**: Contracts with "Active" status
- **Total Value**: Sum of all contract values
- **Pending Approval**: Contracts awaiting approval

### Creating Contracts

1. Go to the **Create Contract** tab
2. Select the project
3. Enter contract details:
   - **Contract Title** (required)
   - **Vendor/Contractor** (required)
   - **Contract Type** (Supply, Labor, Service, etc.)
   - **Contract Value** (required)
   - **Currency** (USD, EUR, GBP, JPY, CNY)
   - **Initial Status** (Draft, Pending, Active, etc.)
4. Add:
   - Description
   - Terms & Conditions
   - Start and End dates
   - Deliverables (one per line)
5. Set up payment schedule:
   - Specify number of payment milestones
   - For each: Description, Amount, Due Date
6. Upload signed contract (PDF)
7. Click **Create Contract**

### Contract Types

- **Supply Agreement**: Equipment and materials
- **Labor Contract**: Installation services
- **Service Agreement**: Ongoing services
- **Maintenance Contract**: System maintenance
- **Consulting Agreement**: Design/engineering services

### Contract Status Flow

Typical contract progression:
1. **Draft**: Initial creation
2. **Pending**: Awaiting approval
3. **Active**: Contract in effect
4. **Completed**: Contract fulfilled
5. **Cancelled**: Contract terminated

### Uploading Templates

1. Go to the **Upload Template** tab
2. Enter template details:
   - Template Name (required)
   - Template Type
   - Description
3. Upload template file (PDF, DOCX, TXT)
4. Click **Upload Template**

### Using Template Library

1. Go to the **Template Library** tab
2. Browse uploaded templates
3. Click **Use Template** to load into contract creation
4. Templates can be deleted if no longer needed

## Portfolio Management

### Portfolio Dashboard

The Portfolio Dashboard provides:
- **Overview Metrics**: Total projects, capacity, budget, active projects
- **Status Distribution**: Pie chart of projects by status
- **Project List**: Detailed table of all projects

### Creating Portfolios

Portfolios are collections of related projects that can be managed together.

To create a portfolio:
1. Projects are automatically available in the portfolio
2. Use filters to view specific project groups
3. Track aggregate metrics across projects

### Portfolio Metrics

Key metrics tracked:
- Total system capacity (kWp)
- Total budget allocation
- Number of projects by status
- Resource utilization
- Contract values

## Tips and Best Practices

### Project Planning

1. **Start with the Wizard**: Use the Project Wizard for consistent project setup
2. **Set Realistic Timelines**: Include buffer time for permitting and weather delays
3. **Document Thoroughly**: Add detailed descriptions and constraints
4. **Track Milestones**: Regular milestone tracking keeps projects on schedule

### Resource Management

1. **Add Resources Early**: Input known resources during project planning
2. **Update Costs Regularly**: Keep unit costs current for accurate budgeting
3. **Track Suppliers**: Maintain supplier information for procurement
4. **Set Availability**: Use availability windows to plan procurement timing
5. **Document Constraints**: Note lead times and payment terms

### Contract Management

1. **Use Templates**: Create reusable templates for standard contracts
2. **Track Deliverables**: List all deliverables for accountability
3. **Payment Schedules**: Tie payments to milestones
4. **Status Updates**: Keep contract status current
5. **File Organization**: Upload and organize signed contracts

### Timeline Planning

1. **Add Phases Early**: Define phases during initial planning
2. **Regular Updates**: Mark milestones complete as achieved
3. **Use Gantt Chart**: Visual timeline helps identify conflicts
4. **Plan Dependencies**: Note which tasks depend on others
5. **Build in Contingency**: Add buffer time for delays

### Data Management

1. **Save Regularly**: Data is auto-saved, but verify after major changes
2. **Backup Data**: Periodically backup the `data/` directory
3. **Organize Files**: Keep uploaded contracts organized
4. **Review Analytics**: Use analytics to identify trends
5. **Clean Old Data**: Archive completed projects periodically

### Performance

1. **Pagination**: For large datasets, filter by project
2. **File Sizes**: Keep uploaded files under 10MB
3. **Regular Cleanup**: Remove unnecessary resources/contracts
4. **Browser Cache**: Clear cache if experiencing issues

## Troubleshooting

### Common Issues

**Issue**: Project not appearing after creation
- **Solution**: Refresh the page or check the Portfolio Dashboard

**Issue**: File upload fails
- **Solution**: Ensure file is under size limit and correct format (PDF, DOCX, TXT)

**Issue**: Gantt chart not displaying
- **Solution**: Ensure project has start/end dates and milestones have valid dates

**Issue**: Data not saving
- **Solution**: Check that `data/` directory exists and is writable

**Issue**: Metrics showing zeros
- **Solution**: Ensure resources/contracts are properly linked to projects

### Getting Help

For additional support:
1. Check the [README.md](../README.md) for setup instructions
2. Review the [API Documentation](API_DOCUMENTATION.md) for technical details
3. Open an issue on GitHub for bugs or feature requests

## Keyboard Shortcuts

Streamlit supports these shortcuts:
- **R**: Rerun the app
- **C**: Clear cache
- **S**: Toggle sidebar (in mobile view)

## Data Privacy

- All data is stored locally in the `data/` directory
- No data is sent to external servers
- Uploaded files are stored in the `uploads/` directory
- You have full control over your data

## Updates and Maintenance

To update the application:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

To back up your data:
```bash
cp -r data/ data_backup_$(date +%Y%m%d)/
cp -r uploads/ uploads_backup_$(date +%Y%m%d)/
```

---

**Need more help?** Check out the full documentation or open an issue on GitHub.
